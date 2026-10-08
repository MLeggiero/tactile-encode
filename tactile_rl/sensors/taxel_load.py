"""Per-taxel normal load on the TaxelScan skins from the hand/tool contacts, batched over environments: the GPU port
of the testbed's elastic-foundation model (tactile_sim.sim.hands.HandIO.skin_taxels and _foundation).

Per patch and physics step (after MuJoCo stepped, on the contacts and poses of that step):

1. the contacts between the patch's collision geoms and the tool with a positive normal force, each kept only if it
   lies within `reach` of some taxel (a contact on a link's side or back does not load the skin); the patch's load is
   their summed normal force (the first constraint row of an elliptic-cone contact);
2. the load's shape, refreshed every ts_shape_dt and whenever the patch starts carrying load: from every taxel within
   ts_shape_radius of the contacts, a ray along its outward normal (starting 3 mm inside the skin) finds the gap to
   the tool; a taxel carries load in proportion to how much less than the skin depth its gap exceeds the smallest one
   (an elastic foundation); with no ray hitting the tool, each contact's force is spread with Gaussian weights
   (sigma ts_spread) about its point instead;
3. the taxels share the load: load x shape.

The tool must be the light hammer's capsule handle (the only tool geom the hand has contact pairs with): its rays are
analytic, mirroring MuJoCo's ray_capsule (engine_ray.c and MuJoCo Warp's ray.py) term for term.

Contacts come from MuJoCo Warp's pooled contact buffer: valid hand/tool contacts are sorted by (world, patch) into at
most `max_per_patch` slots per patch (overflow is counted, never silently dropped), without a host synchronization.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

MJ_MINVAL = float(mujoco.mjMINVAL)
BACK = 0.003  # m: rays start this far inside the skin, so a tool already pressing in is still found


def _quad(a, b, c):
    """MuJoCo's ray_quad: a x^2 + 2 b x + c = 0; (smallest non-negative root or -1, both roots or -1)."""
    det = b * b - a * c
    ok = det >= MJ_MINVAL
    sq = torch.sqrt(torch.clamp(det, min=0.0))
    a_ = torch.where(ok, a, torch.ones_like(a))
    x0 = torch.where(ok, (-b - sq) / a_, -torch.ones_like(a))
    x1 = torch.where(ok, (-b + sq) / a_, -torch.ones_like(a))
    sol = torch.where(x0 >= 0, x0, torch.where(x1 >= 0, x1, -torch.ones_like(x0)))
    return sol, x0, x1


def ray_capsule(pos: torch.Tensor, mat: torch.Tensor, radius: torch.Tensor, half: torch.Tensor, pnt: torch.Tensor,
                vec: torch.Tensor) -> torch.Tensor:
    """Distance along `vec` from `pnt` to a capsule (centre `pos`, rotation `mat`, axis = mat's z column), -1 where the
    ray misses. Shapes broadcast: pos (..., 3), mat (..., 3, 3), radius / half (...), pnt / vec (..., 3)."""
    dif = pnt - pos
    # local frame: mat^T (pnt - pos), mat^T vec
    lpnt = (mat * dif[..., :, None]).sum(-2)
    lvec = (mat * vec[..., :, None]).sum(-2)
    return ray_capsule_local(lpnt, lvec, radius, half)


def ray_capsule_local(lpnt: torch.Tensor, lvec: torch.Tensor, radius: torch.Tensor, half: torch.Tensor
                      ) -> torch.Tensor:
    """ray_capsule with the ray already in the capsule's frame (the bounding-sphere test is rotation invariant)."""
    quad = _quad
    ssz = radius + half
    sol_s, _, _ = quad((lvec * lvec).sum(-1), (lvec * lpnt).sum(-1), (lpnt * lpnt).sum(-1) - ssz * ssz)
    sq_r = radius * radius
    a = lvec[..., 0] ** 2 + lvec[..., 1] ** 2
    b = lvec[..., 0] * lpnt[..., 0] + lvec[..., 1] * lpnt[..., 1]
    c = lpnt[..., 0] ** 2 + lpnt[..., 1] ** 2 - sq_r
    sol, _, _ = quad(a, b, c)
    x = torch.where((sol >= 0) & ((lpnt[..., 2] + sol * lvec[..., 2]).abs() <= half), sol, -torch.ones_like(sol))
    a = a + lvec[..., 2] ** 2
    for side in (1.0, -1.0):  # top cap, then bottom cap: accept only the outer half of each sphere
        ldz = lpnt[..., 2] - side * half
        b = lvec[..., 0] * lpnt[..., 0] + lvec[..., 1] * lpnt[..., 1] + lvec[..., 2] * ldz
        c = lpnt[..., 0] ** 2 + lpnt[..., 1] ** 2 + ldz * ldz - sq_r
        _, x0, x1 = quad(a, b, c)
        for xx in (x0, x1):
            z = lpnt[..., 2] + xx * lvec[..., 2]
            outer = z >= half if side > 0 else z <= -half
            take = (xx >= 0) & outer & ((x < 0) | (xx < x))
            x = torch.where(take, xx, x)
    return torch.where(sol_s >= 0, x, -torch.ones_like(x))


class TaxelLoadModel:
    def __init__(self, parts, mj_model: mujoco.MjModel, num_envs: int, device: str | torch.device = "cpu",
                 prefix: str = "", max_per_patch: int = 8):
        """`parts`: tactile_sim.sim.training_parts.TrainingParts (its `skin`, `tool_geoms`, `handle_geom`); `mj_model`:
        the compiled model the contacts come from, its names prefixed by `prefix` (mjlab: "robot/")."""
        s = parts.cfg.sensors
        self.dev = dev = torch.device(device)
        self.n_env = int(num_envs)
        self.K = int(max_per_patch)
        skin = parts.skin
        if not skin:
            raise ValueError("the scene has no TaxelScan skins")
        m = mj_model
        self.P = P = len(skin)

        def gid(name):
            return m.geom(prefix + name).id

        geom_patch = -np.ones(m.ngeom, dtype=np.int64)
        for k, p in enumerate(skin):
            for g in p.geoms:
                geom_patch[gid(g)] = k
        is_tool = np.zeros(m.ngeom, dtype=bool)
        is_tool[[gid(g) for g in parts.tool_geoms]] = True
        h = gid(parts.handle_geom)
        if m.geom_type[h] != mujoco.mjtGeom.mjGEOM_CAPSULE:
            raise NotImplementedError("the GPU taxel load model ray-casts a capsule handle only")
        self.handle = h
        self.radius = torch.tensor(float(m.geom_size[h, 0]), device=dev)
        self.half = torch.tensor(float(m.geom_size[h, 1]), device=dev)
        self.geom_patch = torch.as_tensor(geom_patch, device=dev)
        self.is_tool = torch.as_tensor(is_tool, device=dev)
        self.body = torch.as_tensor([m.body(prefix + p.body).id for p in skin], device=dev)
        n_of = [len(p.taxel_pos) for p in skin]
        self.T = int(sum(n_of))
        self.patch = torch.as_tensor(np.repeat(np.arange(P), n_of), device=dev)
        f32 = dict(dtype=torch.float32, device=dev)
        self.pos_b = torch.as_tensor(np.concatenate([p.taxel_pos for p in skin]), **f32)
        self.nrm_b = torch.as_tensor(np.concatenate([p.taxel_normal for p in skin]), **f32)
        self.reach2 = torch.as_tensor([p.reach**2 for p in skin], **f32)
        start = np.concatenate([[0], np.cumsum(n_of)[:-1]])
        self.groups = []  # patches with the same taxel count, computed together
        self.group_taxels = []  # per group: taxel positions, ray origins, normals (p, n, 3), |pos|^2 (p, n)
        for n in sorted(set(n_of)):
            ks = [k for k in range(P) if n_of[k] == n]
            idx = np.stack([np.arange(start[k], start[k] + n) for k in ks])
            self.groups.append((torch.as_tensor(ks, device=dev), torch.as_tensor(idx, device=dev)))
            pos_g = self.pos_b[torch.as_tensor(idx, device=dev)]
            nrm_g = self.nrm_b[torch.as_tensor(idx, device=dev)]
            self.group_taxels.append((pos_g, pos_g - BACK * nrm_g, nrm_g, (pos_g * pos_g).sum(-1)))
        self.shape_r2 = float(s.ts_shape_radius) ** 2
        self.sigma2 = float(s.ts_spread) ** 2
        self.shape_dt = float(s.ts_shape_dt)
        N = self.n_env
        self.skin_depth = torch.full((N,), float(s.ts_skin_depth), **f32)
        self.reset_state()

    def reset_state(self) -> None:
        N, P, T, dev = self.n_env, self.P, self.T, self.dev
        self.has_shape = torch.zeros(N, P, dtype=torch.bool, device=dev)
        self.fallback = torch.zeros(N, P, dtype=torch.bool, device=dev)
        self.t_shape = torch.full((N, P), -1e9, dtype=torch.float64, device=dev)
        self.shape = torch.zeros(N, T, dtype=torch.float32, device=dev)
        self.taxels = torch.zeros(N, T, dtype=torch.float32, device=dev)
        self.total = torch.zeros(N, P, dtype=torch.float32, device=dev)
        self.overflow = torch.zeros((), dtype=torch.long, device=dev)
        self.contact_f = torch.zeros(N, P, self.K, dtype=torch.float32, device=dev)
        self.contact_pos = torch.zeros(N, P, self.K, 3, dtype=torch.float32, device=dev)
        self.contact_ok = torch.zeros(N, P, self.K, dtype=torch.bool, device=dev)

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.has_shape[ids] = False
        self.fallback[ids] = False
        self.t_shape[ids] = -1e9
        self.taxels[ids] = 0.0
        self.total[ids] = 0.0

    # ------------------------------------------------------------------ contacts
    def dense_contacts(self, data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The hand/tool contacts of a MuJoCo Warp step, (N, P, K) slots: normal force, world position, occupied."""
        N, P, K, dev = self.n_env, self.P, self.K, self.dev
        con = data.contact
        geom = con.geom[:]
        C = geom.shape[0]
        g1, g2 = geom[:, 0].long(), geom[:, 1].long()
        adr = con.efc_address[:]
        adr = (adr[:, 0] if adr.ndim > 1 else adr).long()
        wid = con.worldid[:].long()
        valid = torch.arange(C, device=dev) < data.nacon[:].long()
        t1, t2 = self.is_tool[g1], self.is_tool[g2]
        patch = self.geom_patch[torch.where(t1, g2, g1)]
        ok = valid & (adr >= 0) & (t1 ^ t2) & (patch >= 0)
        w = wid.clamp(0, N - 1)
        force = data.efc.force[:][w, adr.clamp(min=0)]
        ok = ok & (force > 0)
        key = torch.where(ok, w * P + patch, torch.full_like(w, N * P))
        skey, order = torch.sort(key, stable=True)
        rank = torch.arange(C, device=dev) - torch.searchsorted(skey, skey)
        keep = (skey < N * P) & (rank < K)
        self.overflow += ((skey < N * P) & (rank >= K)).sum()
        dest = torch.where(keep, skey * K + rank, torch.full_like(skey, N * P * K))
        f = torch.zeros(N * P * K + 1, dtype=torch.float32, device=dev)
        f.scatter_(0, dest, force[order].float())
        pos = torch.zeros(N * P * K + 1, 3, dtype=torch.float32, device=dev)
        pos.scatter_(0, dest[:, None].expand(-1, 3), con.pos[:][order].float())
        occ = torch.zeros(N * P * K + 1, dtype=torch.bool, device=dev)
        occ.scatter_(0, dest, keep)
        return f[:-1].view(N, P, K), pos[:-1].view(N, P, K, 3), occ[:-1].view(N, P, K)

    # ------------------------------------------------------------------ loads
    def step(self, data, t: torch.Tensor) -> torch.Tensor:
        """One physics step from MuJoCo Warp data (after the step); `t` (N,) float64 the time now. Returns (N, n_taxels)
        loads (N) in layout order."""
        f, pos, occ = self.dense_contacts(data)
        xpos = data.xpos[:][:, self.body]
        xmat = data.xmat[:][:, self.body].reshape(self.n_env, self.P, 3, 3)
        cpos = data.geom_xpos[:][:, self.handle]
        cmat = data.geom_xmat[:][:, self.handle].reshape(self.n_env, 3, 3)
        return self.loads(f, pos, occ, xpos, xmat, cpos, cmat, t)

    def loads(self, f, pos, occ, xpos, xmat, cpos, cmat, t: torch.Tensor) -> torch.Tensor:
        """The model on given contacts (N, P, K), patch body poses (N, P, 3) / (N, P, 3, 3) and the capsule's pose
        (N, 3) / (N, 3, 3); see the module doc."""
        N, P, dev = self.n_env, self.P, self.dev
        f, pos, xpos, xmat, cpos, cmat = (x.float() for x in (f, pos, xpos, xmat, cpos, cmat))
        out = torch.zeros(N, self.T, dtype=torch.float32, device=dev)
        total = torch.zeros(N, P, dtype=torch.float32, device=dev)
        new_shape = torch.zeros(N, self.T, dtype=torch.float32, device=dev)
        new_fallback = torch.zeros(N, P, dtype=torch.bool, device=dev)
        gauss = torch.zeros(N, self.T, dtype=torch.float32, device=dev)
        loaded = torch.zeros(N, P, dtype=torch.bool, device=dev)
        reach_all = torch.zeros(N, P, self.K, dtype=torch.bool, device=dev)
        gap = torch.zeros(N, self.T, dtype=torch.float32, device=dev)
        for (ks, idx), (pos_g, org_g, nrm_g, pb2) in zip(self.groups, self.group_taxels, strict=True):
            Rg, xg = xmat[:, ks], xpos[:, ks]  # (N, p, 3, 3), (N, p, 3)
            # rays from each taxel along its normal, mapped into the handle capsule's frame
            o = torch.einsum("npij,pkj->npki", Rg, org_g) + xg[:, :, None]  # (N, p, n, 3) world
            v = torch.einsum("npij,pkj->npki", Rg, nrm_g)
            lpnt = torch.einsum("nji,npkj->npki", cmat, o - cpos[:, None, None])
            lvec = torch.einsum("nji,npkj->npki", cmat, v)
            hit = ray_capsule_local(lpnt, lvec, self.radius, self.half)
            g = torch.where(hit >= 0, hit - BACK, torch.full_like(hit, float("inf")))  # (N, p, n)
            gap[:, idx] = g
            # the contacts in each patch body's frame, squared distances to its taxels (N, p, n, K)
            q = torch.einsum("npji,npkj->npki", Rg, pos[:, ks] - xg[:, :, None])  # (N, p, K, 3)
            d2 = pb2[None, :, :, None] + (q * q).sum(-1)[:, :, None, :] - \
                2.0 * torch.einsum("ptj,npkj->nptk", pos_g, q)
            d2 = d2.clamp(min=0.0)
            d2min = d2.min(2).values  # (N, p, K)
            reach = occ[:, ks] & (d2min <= self.reach2[ks][None, :, None])
            fk = torch.where(reach, f[:, ks], torch.zeros_like(f[:, ks]))
            reach_all[:, ks] = reach
            total[:, ks] = fk.sum(-1)
            loaded[:, ks] = reach.any(-1)
            # Gaussian spread of each contact about its point (the fallback)
            wts = torch.exp(-0.5 * (d2 - d2min[:, :, None, :]) / self.sigma2)
            wts = wts / wts.sum(2, keepdim=True).clamp(min=1e-30)
            gauss[:, idx] = (wts * fk[:, :, None, :]).sum(-1)
            # elastic foundation over the taxels near the contacts
            near = (reach[:, :, None, :] & (d2 <= self.shape_r2)).any(-1)  # (N, p, n)
            fin = near & torch.isfinite(g)
            gmin = torch.where(fin, g, torch.full_like(g, float("inf"))).min(-1).values  # (N, p)
            w = torch.clamp(self.skin_depth[:, None, None] - (g - gmin[..., None]), min=0.0)
            w = torch.where(fin, w, torch.zeros_like(w))
            sw = w.sum(-1)
            new_shape[:, idx] = w / sw.clamp(min=1e-30)[..., None]
            new_fallback[:, ks] = sw <= 0
        refresh = loaded & (~self.has_shape | (t[:, None] - self.t_shape >= self.shape_dt - 1e-12))
        r_t = refresh[:, self.patch]
        self.shape = torch.where(r_t, new_shape, self.shape)
        self.fallback = torch.where(refresh, new_fallback, self.fallback)
        self.t_shape = torch.where(refresh, t[:, None], self.t_shape)
        self.has_shape = loaded
        load_t = loaded[:, self.patch]
        out = torch.where(load_t, torch.where(self.fallback[:, self.patch], gauss, total[:, self.patch] * self.shape),
                          torch.zeros_like(out))
        self.taxels, self.total = out, total
        self.contact_f, self.contact_pos, self.contact_ok = f, pos, reach_all
        return out
