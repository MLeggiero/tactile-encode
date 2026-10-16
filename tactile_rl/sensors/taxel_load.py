"""Per-taxel normal load on the TaxelScan skins from the hand/tool contacts, batched over environments: the GPU port
of the testbed's elastic-foundation model (tactile_sim.sim.hands.HandIO.skin_taxels and _foundation).

Per patch and physics step (after MuJoCo stepped, on the contacts and poses of that step):

1. the contacts between the patch's collision geoms and the tool with a positive normal force, each kept only if it
   lies within `reach` of some taxel (a contact on a link's side or back does not load the skin); the patch's load is
   their summed normal force (the first constraint row of an elliptic-cone contact);
2. the load's shape, refreshed every ts_shape_dt, whenever the patch starts carrying load and whenever the set of tool
   geoms loading it changes (as the testbed's): from every taxel within ts_shape_radius of the contacts, a ray along
   its outward normal (starting 3 mm inside the skin) finds the gap to the tool geoms the patch has contact pairs
   with; a taxel carries load in proportion to how much less than the skin depth its gap exceeds the smallest one (an
   elastic foundation); with no ray hitting the tool, each contact's force is spread with Gaussian weights (sigma
   ts_spread) about its point instead;
3. the taxels share the load: load x shape.

The tool's ray geoms are analytic:
- a capsule handle (the light hammer): MuJoCo's ray_capsule (engine_ray.c and MuJoCo Warp's ray.py) term for term, in
  torch;
- convex mesh hulls (the YCB claw hammer's handle, a chain of convex slices; any mesh whose faces are its convex
  hull, as MuJoCo makes them from a vertex cloud): the ray's entry into the intersection of the hull's face planes
  (Cyrus-Beck), which is where mj_rayMesh meets the hull's first triangle (its exit, for a ray starting inside), in a
  Warp kernel that only casts the rays a refresh needs and skips a hull whose bounding sphere the ray misses (the
  testbed's cut);
- a mix (the DF333D: its grip's two capsules and the housing's, the trigger a box): the capsules as the light hammer's,
  a box as the hull of its six faces in the same kernel (where mju_rayGeom's ray_box meets the same face), the nearest
  hit over the geoms the patch pairs with.

Contacts come from MuJoCo Warp's pooled contact buffer: valid hand/tool contacts are sorted by (world, patch) into at
most `max_per_patch` slots per patch (overflow is counted, never silently dropped), without a host synchronization.
The stateless parts (contacts, distances, the foundation's weights, the capsule's rays) can run through torch.compile
(`compiled`), which fuses their elementwise chains; the shape-refresh bookkeeping stays eager.
"""

from __future__ import annotations

import math

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


# ---------------------------------------------------------------------------------------------- convex hulls


def hull_planes(m: mujoco.MjModel, geom_id: int, tol: float = 1e-9) -> np.ndarray:
    """The face planes (n, h) of a mesh geom whose faces bound a convex solid (n . x <= h inside; geom frame), one per
    distinct plane (coplanar triangles merged), float64 (F, 4)."""
    mid = int(m.geom_dataid[geom_id])
    v0, nv = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
    f0, nf = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
    V = np.asarray(m.mesh_vert[v0:v0 + nv], float)
    F = np.asarray(m.mesh_face[f0:f0 + nf], int)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1)
    keep = ln > 1e-14
    n = n[keep] / ln[keep, None]
    h = np.einsum("ij,ij->i", n, a[keep])
    centre = V.mean(axis=0)
    flip = np.einsum("ij,j->i", n, centre) > h  # outward: the centre lies inside
    n[flip], h[flip] = -n[flip], -h[flip]
    planes = np.concatenate([n, h[:, None]], axis=1)
    out: list[np.ndarray] = []
    for p in planes[np.lexsort(planes.T[::-1])]:
        if not out or np.max(np.abs(out[-1] - p)) > tol:
            out.append(p)
    planes = np.array(out)
    # every vertex inside or on every plane: the faces bound a convex solid
    viol = V @ planes[:, :3].T - planes[:, 3][None]
    if viol.max() > 1e-6:
        raise ValueError(f"mesh geom {m.geom(geom_id).name!r} is not convex ({viol.max():.2e} m outside a face)")
    return planes


def box_planes(m: mujoco.MjModel, geom_id: int) -> np.ndarray:
    """A box geom's six face planes (n, h) in its frame (n . x <= h inside), float64 (6, 4)."""
    size = np.asarray(m.geom_size[geom_id], float)
    return np.array([[*(s * np.eye(3)[i]), size[i]] for i in range(3) for s in (1.0, -1.0)])


_KERNEL = None


def _hull_kernel():
    """The Warp kernel (built once): per (world, taxel) the nearest gap to the hulls its patch pairs with."""
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import warp as wp

    wp.init()  # idempotent (MuJoCo Warp has usually done it)

    @wp.kernel
    def ray_hulls(orig: wp.array2d(dtype=wp.vec3), dirs: wp.array2d(dtype=wp.vec3), need: wp.array2d(dtype=wp.int32),
                  taxel_patch: wp.array(dtype=wp.int32), patch_tool: wp.array2d(dtype=wp.int32),
                  gpos: wp.array2d(dtype=wp.vec3), gmat: wp.array2d(dtype=wp.mat33),
                  planes: wp.array2d(dtype=wp.vec4), nplanes: wp.array(dtype=wp.int32),
                  rbound: wp.array(dtype=wp.float32), back: wp.float32, n_geom: wp.int32,
                  gap: wp.array2d(dtype=wp.float32)):
        n, t = wp.tid()
        best = wp.float32(1.0e30)
        if need[n, t] != 0:
            o = orig[n, t]
            v = dirs[n, t]
            k = taxel_patch[t]
            for g in range(n_geom):
                if patch_tool[k, g] != 0:
                    c = gpos[n, g]
                    dc = c - o
                    tt = wp.dot(dc, v)
                    perp2 = wp.dot(dc, dc) - tt * tt
                    r = rbound[g] * wp.float32(1.000001) + wp.float32(1.0e-9)
                    if perp2 <= r * r and tt >= -r:
                        R = gmat[n, g]
                        lp = wp.transpose(R) * (o - c)
                        ld = wp.transpose(R) * v
                        t_in = wp.float32(-1.0e30)
                        t_out = wp.float32(1.0e30)
                        ok = wp.int32(1)
                        for f in range(nplanes[g]):
                            pl = planes[g, f]
                            nrm = wp.vec3(pl[0], pl[1], pl[2])
                            den = wp.dot(nrm, ld)
                            dist = pl[3] - wp.dot(nrm, lp)
                            if wp.abs(den) < wp.float32(1.0e-12):
                                if dist < wp.float32(0.0):
                                    ok = wp.int32(0)
                            else:
                                tp = dist / den
                                if den < wp.float32(0.0):
                                    t_in = wp.max(t_in, tp)
                                else:
                                    t_out = wp.min(t_out, tp)
                        if ok != 0 and t_in <= t_out:
                            hit = wp.float32(-1.0)
                            if t_in >= wp.float32(0.0):
                                hit = t_in
                            elif t_out >= wp.float32(0.0):
                                hit = t_out
                            if hit >= wp.float32(0.0):
                                best = wp.min(best, hit - back)
        gap[n, t] = best

    _KERNEL = ray_hulls
    return _KERNEL


class TaxelLoadModel:
    def __init__(self, parts, mj_model: mujoco.MjModel, num_envs: int, device: str | torch.device = "cpu",
                 prefix: str = "", max_per_patch: int | None = None, compiled: bool = False):
        """`parts`: tactile_sim.sim.training_parts.TrainingParts (its `skin`, `tool_geoms`); `mj_model`: the compiled
        model the contacts come from, its names prefixed by `prefix` (mjlab: "robot/"). `max_per_patch` contact slots
        per patch; None: 4 against a capsule handle (MuJoCo Warp makes one contact per convex pair, a distal pad has
        two hulls), 12 against hull slices or a mix of geoms (a palm along the handle touches several)."""
        s = parts.cfg.sensors
        self.dev = dev = torch.device(device)
        self.n_env = int(num_envs)
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
        # the tool geoms each patch has contact pairs with: the ones its rays test (HandIO._pair_tool)
        pair_tool: list[set[int]] = [set() for _ in range(P)]
        for a, b in zip(m.pair_geom1, m.pair_geom2, strict=True):
            a, b = int(a), int(b)
            for hg, tg in ((a, b), (b, a)):
                if geom_patch[hg] >= 0 and is_tool[tg] and not is_tool[hg]:
                    pair_tool[int(geom_patch[hg])].add(tg)
        ray_geoms = sorted(set().union(*pair_tool))
        if not ray_geoms:
            raise ValueError("no skin patch has contact pairs with the tool")
        self.ray_geoms = torch.as_tensor(ray_geoms, dtype=torch.long, device=dev)
        self.G = G = len(ray_geoms)
        types = [int(m.geom_type[g]) for g in ray_geoms]
        CAPSULE, BOX, MESH = (int(t) for t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_BOX,
                                               mujoco.mjtGeom.mjGEOM_MESH))
        if set(types) - {CAPSULE, BOX, MESH}:
            raise NotImplementedError(f"the GPU taxel load model ray-casts capsules, boxes and convex mesh hulls, not "
                                      f"geom types {sorted(set(types))}")
        self.caps: list[tuple[int, float, float]] = []  # the mixed mode's capsules: (ray index, radius, half length)
        if types == [CAPSULE]:
            self.mode = "capsule"
            h = ray_geoms[0]
            self.radius = torch.tensor(float(m.geom_size[h, 0]), device=dev)
            self.half = torch.tensor(float(m.geom_size[h, 1]), device=dev)
        else:
            # "hull": convex meshes only; "mixed": capsules too (cast in torch) and boxes (hulls in the kernel)
            self.mode = "hull" if set(types) == {MESH} else "mixed"
            flat = [i for i, ty in enumerate(types) if ty != CAPSULE]
            self.caps = [(i, float(m.geom_size[ray_geoms[i], 0]), float(m.geom_size[ray_geoms[i], 1]))
                         for i, ty in enumerate(types) if ty == CAPSULE]
            pl = [hull_planes(m, ray_geoms[i]) if types[i] == MESH else box_planes(m, ray_geoms[i]) for i in flat]
            Fm = max((len(p) for p in pl), default=1)
            planes = np.zeros((len(pl), Fm, 4))
            planes[..., 3] = 1.0  # padding: 0 . x <= 1, never binding
            for i, p in enumerate(pl):
                planes[i, :len(p)] = p
            self.flat = torch.as_tensor(flat, dtype=torch.long, device=dev)  # the kernel's geoms among the ray geoms
            self.planes = torch.as_tensor(planes, dtype=torch.float32, device=dev)
            self.nplanes = torch.as_tensor([len(p) for p in pl], dtype=torch.int32, device=dev)
            self.rbound = torch.as_tensor([float(m.geom_rbound[ray_geoms[i]]) for i in flat], dtype=torch.float32,
                                          device=dev)
        self.K = int(max_per_patch) if max_per_patch else (4 if self.mode == "capsule" else 12)
        ray_index = -np.ones(m.ngeom, dtype=np.int64)
        ray_index[ray_geoms] = np.arange(G)
        self.ray_index = torch.as_tensor(ray_index, device=dev)
        pt = np.zeros((P, G), dtype=np.int32)
        for k in range(P):
            for g in pair_tool[k]:
                pt[k, ray_index[g]] = 1
        self.patch_tool = torch.as_tensor(pt, device=dev)
        self.geom_patch = torch.as_tensor(geom_patch, device=dev)
        self.is_tool = torch.as_tensor(is_tool, device=dev)
        self.body = torch.as_tensor([m.body(prefix + p.body).id for p in skin], device=dev)
        n_of = [len(p.taxel_pos) for p in skin]
        self.T = int(sum(n_of))
        self.patch = torch.as_tensor(np.repeat(np.arange(P), n_of), device=dev)
        self.patch_i32 = self.patch.to(torch.int32)
        if self.mode != "capsule":
            self.flat_tool = self.patch_tool[:, self.flat].contiguous()  # the pairs with the kernel's geoms (P, Gf)
            self.cap_pair = [self.patch_tool[self.patch, i].bool() for i, _, _ in self.caps]  # per capsule (T,)
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
        self.compiled = bool(compiled)  # torch.compile the stateless core (the env does; ~50x fewer kernels)
        self._contacts_c = torch.compile(self._contacts, dynamic=False) if self.compiled else None
        self._shape_c = torch.compile(self._shape, dynamic=False) if self.compiled else None
        self._wp_stream = None
        self.reset_state()

    def reset_state(self) -> None:
        N, P, T, G, dev = self.n_env, self.P, self.T, self.G, self.dev
        self.has_shape = torch.zeros(N, P, dtype=torch.bool, device=dev)
        self.fallback = torch.zeros(N, P, dtype=torch.bool, device=dev)
        self.t_shape = torch.full((N, P), -1e9, dtype=torch.float64, device=dev)
        self.shape_geoms = torch.zeros(N, P, G, dtype=torch.bool, device=dev)  # the tool geoms at the last refresh
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
        self.shape_geoms[ids] = False
        self.taxels[ids] = 0.0
        self.total[ids] = 0.0

    # ------------------------------------------------------------------ contacts
    def dense_contacts(self, data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The hand/tool contacts of a MuJoCo Warp step, (N, P, K) slots: normal force, world position, occupied, and
        the tool geom (its index among the ray geoms, -1 for another tool geom)."""
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
        tg = self.ray_index[torch.where(t1, g1, g2)]
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
        gi = torch.full((N * P * K + 1,), -1, dtype=torch.long, device=dev)
        gi.scatter_(0, dest, tg[order])
        return f[:-1].view(N, P, K), pos[:-1].view(N, P, K, 3), occ[:-1].view(N, P, K), gi[:-1].view(N, P, K)

    # ------------------------------------------------------------------ loads
    def step(self, data, t: torch.Tensor) -> torch.Tensor:
        """One physics step from MuJoCo Warp data (after the step); `t` (N,) float64 the time now. Returns (N, n_taxels)
        loads (N) in layout order."""
        f, pos, occ, tg = self.dense_contacts(data)
        xpos = data.xpos[:][:, self.body]
        xmat = data.xmat[:][:, self.body].reshape(self.n_env, self.P, 3, 3)
        gpos = data.geom_xpos[:][:, self.ray_geoms]
        gmat = data.geom_xmat[:][:, self.ray_geoms].reshape(self.n_env, self.G, 3, 3)
        return self.loads(f, pos, occ, xpos, xmat, gpos, gmat, t, tg)

    def loads(self, f, pos, occ, xpos, xmat, gpos, gmat, t: torch.Tensor, tg: torch.Tensor | None = None
              ) -> torch.Tensor:
        """The model on given contacts (N, P, K), patch body poses (N, P, 3) / (N, P, 3, 3), the ray geoms' poses
        (N, G, 3) / (N, G, 3, 3) (a capsule's may come as (N, 3) / (N, 3, 3)) and the contacts' tool geoms (N, P, K)
        (indices among the ray geoms; None: the shape does not follow changes of the touching geoms); see the module
        doc."""
        f, pos, xpos, xmat, gpos, gmat = (x.float() for x in (f, pos, xpos, xmat, gpos, gmat))
        if gpos.ndim == 2:
            gpos, gmat = gpos[:, None], gmat[:, None]
        contacts = self._contacts_c if self.compiled else self._contacts
        total, loaded, reach_all, gauss, near, org, ray = contacts(f, pos, occ, xpos, xmat)
        due = ~self.has_shape | (t[:, None] - self.t_shape >= self.shape_dt - 1e-12)
        geoms = torch.zeros(self.n_env, self.P, self.G, dtype=torch.bool, device=self.dev)
        if tg is not None:
            onehot = (tg[..., None] == torch.arange(self.G, device=self.dev)) & reach_all[..., None]
            geoms = onehot.any(2)
            due = due | (geoms != self.shape_geoms).any(-1)
        refresh = loaded & due
        r_t = refresh[:, self.patch]
        if self.mode == "capsule":
            gap = None
        elif len(self.flat):
            gap = self._hull_gaps(org, ray, near & r_t, gpos, gmat)
        else:
            gap = torch.full_like(org[..., 0], math.inf)
        shape_fn = self._shape_c if self.compiled else self._shape
        new_shape, new_fallback = shape_fn(near, org, ray, gpos, gmat, self.skin_depth, gap)
        self.shape = torch.where(r_t, new_shape, self.shape)
        self.fallback = torch.where(refresh, new_fallback, self.fallback)
        self.t_shape = torch.where(refresh, t[:, None], self.t_shape)
        self.shape_geoms = torch.where(refresh[..., None], geoms, self.shape_geoms)
        self.has_shape = loaded
        load_t = loaded[:, self.patch]
        out = torch.where(load_t, torch.where(self.fallback[:, self.patch], gauss, total[:, self.patch] * self.shape),
                          torch.zeros_like(gauss))
        self.taxels, self.total = out, total
        self.contact_f, self.contact_pos, self.contact_ok = f, pos, reach_all
        return out

    def _contacts(self, f, pos, occ, xpos, xmat):
        """The stateless contact part: per patch the load and the contacts that count, the Gaussian spread, the taxels
        near the contacts and every taxel's ray (origin, direction; world). A pure function of its inputs."""
        N, P, dev = f.shape[0], self.P, f.device
        total = torch.zeros(N, P, dtype=torch.float32, device=dev)
        gauss = torch.zeros(N, self.T, dtype=torch.float32, device=dev)
        near_all = torch.zeros(N, self.T, dtype=torch.bool, device=dev)
        org = torch.zeros(N, self.T, 3, dtype=torch.float32, device=dev)
        ray = torch.zeros(N, self.T, 3, dtype=torch.float32, device=dev)
        loaded = torch.zeros(N, P, dtype=torch.bool, device=dev)
        reach_all = torch.zeros(N, P, f.shape[2], dtype=torch.bool, device=dev)
        for (ks, idx), (pos_g, org_g, nrm_g, pb2) in zip(self.groups, self.group_taxels, strict=True):
            Rg, xg = xmat[:, ks], xpos[:, ks]  # (N, p, 3, 3), (N, p, 3)
            # rays from each taxel along its normal (world)
            org[:, idx] = torch.einsum("npij,pkj->npki", Rg, org_g) + xg[:, :, None]  # (N, p, n, 3)
            ray[:, idx] = torch.einsum("npij,pkj->npki", Rg, nrm_g)
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
            # the taxels near the contacts: the foundation's
            near_all[:, idx] = (reach[:, :, None, :] & (d2 <= self.shape_r2)).any(-1)  # (N, p, n)
        return total, loaded, reach_all, gauss, near_all, org, ray

    def _shape(self, near, org, ray, gpos, gmat, skin_depth, gap):
        """A fresh foundation shape (with whether it found the tool) from the taxels near the contacts and their gaps
        to the tool (`gap` (N, T), inf where no ray hits; None: the capsule's rays, cast here; a mix's capsules are
        cast here too)."""
        N, P, dev = near.shape[0], self.P, near.device
        if gap is None:
            cpos, cmat = gpos[:, 0], gmat[:, 0]
            lpnt = torch.einsum("nji,ntj->nti", cmat, org - cpos[:, None])
            lvec = torch.einsum("nji,ntj->nti", cmat, ray)
            hit = ray_capsule_local(lpnt, lvec, self.radius, self.half)
            gap = torch.where(hit >= 0, hit - BACK, torch.full_like(hit, float("inf")))
        elif self.caps:
            gap = torch.minimum(gap, self._capsule_gaps(org, ray, gpos, gmat))
        new_shape = torch.zeros(N, self.T, dtype=torch.float32, device=dev)
        new_fallback = torch.zeros(N, P, dtype=torch.bool, device=dev)
        for (ks, idx), _ in zip(self.groups, self.group_taxels, strict=True):
            g = gap[:, idx]  # (N, p, n)
            nr = near[:, idx]
            fin = nr & torch.isfinite(g)
            gmin = torch.where(fin, g, torch.full_like(g, float("inf"))).min(-1).values  # (N, p)
            w = torch.clamp(skin_depth[:, None, None] - (g - gmin[..., None]), min=0.0)
            w = torch.where(fin, w, torch.zeros_like(w))
            sw = w.sum(-1)
            new_shape[:, idx] = w / sw.clamp(min=1e-30)[..., None]
            new_fallback[:, ks] = sw <= 0
        return new_shape, new_fallback

    def _capsule_gaps(self, org, ray, gpos, gmat) -> torch.Tensor:
        """(N, T) gaps from every taxel's ray to the capsules of a mix its patch pairs with (inf: no hit); `gpos` /
        `gmat` are all the ray geoms' poses."""
        gap = torch.full(org.shape[:2], math.inf, dtype=org.dtype, device=org.device)
        for (i, radius, half), pair in zip(self.caps, self.cap_pair, strict=True):
            lpnt = torch.einsum("nji,ntj->nti", gmat[:, i], org - gpos[:, i, None])
            lvec = torch.einsum("nji,ntj->nti", gmat[:, i], ray)
            hit = ray_capsule_local(lpnt, lvec, radius, half)
            gap = torch.where((hit >= 0) & pair, torch.minimum(gap, hit - BACK), gap)
        return gap

    def _hull_gaps(self, org, ray, need, gpos, gmat) -> torch.Tensor:
        """(N, T) gaps from each needed taxel's ray to the hulls (and a mix's boxes) its patch pairs with (inf: none
        needed or no hit); `gpos` / `gmat` are all the ray geoms' poses."""
        import warp as wp

        kernel = _hull_kernel()  # built (and Warp initialized) once
        N, T = self.n_env, self.T
        if self.mode == "mixed":
            gpos, gmat = gpos[:, self.flat], gmat[:, self.flat]
        out = torch.empty(N, T, dtype=torch.float32, device=self.dev)
        args = [wp.from_torch(org.contiguous(), dtype=wp.vec3), wp.from_torch(ray.contiguous(), dtype=wp.vec3),
                wp.from_torch(need.to(torch.int32).contiguous()), wp.from_torch(self.patch_i32),
                wp.from_torch(self.flat_tool), wp.from_torch(gpos.contiguous(), dtype=wp.vec3),
                wp.from_torch(gmat.contiguous(), dtype=wp.mat33), wp.from_torch(self.planes, dtype=wp.vec4),
                wp.from_torch(self.nplanes), wp.from_torch(self.rbound), float(BACK), int(len(self.flat))]
        kw = {}
        if self.dev.type == "cuda":  # on torch's stream, ordered with the tensors it reads and writes
            ts = torch.cuda.current_stream(self.dev)
            if self._wp_stream is None or self._wp_stream[0] != ts.cuda_stream:
                self._wp_stream = (ts.cuda_stream, wp.stream_from_torch(ts))
            kw["stream"] = self._wp_stream[1]
        dev = self.dev
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        wp.launch(kernel, dim=(N, T), inputs=args, outputs=[wp.from_torch(out)],
                  device=wp.device_from_torch(dev), **kw)
        return torch.where(out < 1.0e29, out, torch.full_like(out, math.inf))
