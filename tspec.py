### Trispectrum estimator (parity-even, isotropic ell=0) for 3D periodic-box fields.
### Extension of PolyBin3D. 
### Huy 2026

import numpy as np


class TSpec:
    """Binned parity-even trispectrum estimator.

    Inputs:
    - base      : PolyBin3D base instance (FFTs, MapUtils, applySinv).
    - k_bins    : 1D array of side k-bin edges, length n_k+1.
    - diag_bins : 1D array of internal-diagonal K-bin edges (coarse), length n_diag+1.
    - shapes    : list of (i, j, k, l) side-bin index tuples (reduced family).
    - applySinv : weighting function; default base.applySinv_trivial (ideal estimator).
    """

    def __init__(self, base, k_bins, diag_bins, shapes, applySinv=None, pair_chunk=16):
        self.base = base
        self.k_bins = np.asarray(k_bins)
        self.diag_bins = np.asarray(diag_bins)
        self.shapes = list(shapes)
        self.applySinv = applySinv if applySinv is not None else base.applySinv_trivial
   
        self.pair_chunk = int(pair_chunk)

        self.n_k = len(self.k_bins) - 1
        self.n_diag = len(self.diag_bins) - 1
        self.shapes_odd = self.odd_shapes(self.n_k)
        self.shapes_even = self.even_shapes(self.n_k)   

    # ------------------------------------------------------------------
    # band fitter
    # ------------------------------------------------------------------
    def _as_input(self, delta, input_type):
        """To make sure delta from jax is usable by numpy FFT 
        """
        if input_type == 'real' and getattr(self.base, 'backend', None) != 'jax':
            return np.asarray(delta, dtype=np.float64)
        return delta

    def _process_sim(self, delta, input_type='real'):
        """Return shell maps g_b(x) for every side bin b.

         g_b(x) = IFFT[ W_b(k) * delta(k) ],  W_b = top-hat shell.

        Returns: list of n_k real-space arrays.
        """
        delta_f = self.applySinv(self._as_input(delta, input_type),
                                 input_type=input_type, output_type='fourier')
        g_b_list = []
        for b in range(self.n_k):
            filtered = self.base.map_utils.fourier_filter(delta_f, 0, self.k_bins[b], self.k_bins[b+1])
            g_b_list.append(self.base.to_real(filtered))
        return g_b_list

    # ------------------------------------------------------------------
    # quartic product (raw numerator)
    # ------------------------------------------------------------------
    def _pair_field(self, g_i, g_j):
        """Real-space pair field P_ij(x) = g_i(x) * g_j(x)."""
        return self.base.map_utils.prod_real(g_i, g_j)

    def _diag_filter(self, P_real, B):
        """Filter a real pair field to internal-diagonal K-bin B.
        P^B_ij(x) = IFFT[ W_B(K) * FFT[P_ij](K) ].
        """

        P_f = self.base.to_fourier(P_real) 
        filt = self.base.map_utils.fourier_filter(P_f, 0, self.diag_bins[B], self.diag_bins[B+1]) 
        return self.base.to_real(filt)

    def _numerator(self, fw, fx, fy, fz, shapes=None):
        """General quartic contraction Q over four shell-map-lists.

        Q[s=(i,j,k,l), B] = sum_x diag_B(fw_i * fx_j)(x) * (fy_k * fz_l)(x).

        Each of fw, fx, fy, fz is a list of n_k real-space shell maps.
    
        """
        shapes = self.shapes if shapes is None else list(shapes)
        if not shapes:
            return np.zeros((0, self.n_diag))

        pairs12 = sorted({(i, j) for i, j, _, _ in shapes})
        pairs34 = sorted({(k, l) for _, _, k, l in shapes})
        idx12 = {p: n for n, p in enumerate(pairs12)}
        idx34 = {p: n for n, p in enumerate(pairs34)}
        rows = np.array([idx12[(i, j)] for i, j, _, _ in shapes])
        cols = np.array([idx34[(k, l)] for _, _, k, l in shapes])

        Pf12 = [self.base.to_fourier(self._pair_field(fw[i], fx[j])) for i, j in pairs12] #1st triangle in Fourrier space
        
        M34 = np.stack([np.asarray(self._pair_field(fy[k], fz[l])).ravel() #2nd triangle but in real space
                        for k, l in pairs34])

        out = np.empty((len(shapes), self.n_diag))
        G = np.empty((len(pairs12), len(pairs34)))
        for B in range(self.n_diag):  #loop over K shells
            lo, hi = self.diag_bins[B], self.diag_bins[B+1]
            for c0 in range(0, len(pairs12), self.pair_chunk):
                #amplitude of 1st triangle filtered to the K shell
                blk = np.stack([np.asarray(self.base.to_real(
                                    self.base.map_utils.fourier_filter(Pf, 0, lo, hi))).ravel() 
                                for Pf in Pf12[c0:c0+self.pair_chunk]])
                G[c0:c0+len(blk)] = blk @ M34.T #glued 1st and 2nd triangle together, sum over x
            out[:, B] = G[rows, cols]
        return out

    def Tk_numerator(self, delta, input_type='real'):
        """Raw 4-field quartic numerator (NO disconnected subtraction).

        Returns: array (n_shapes, n_diag), t[s,B] = Q(g,g,g,g)[s,B].
        """
        g = self._process_sim(delta, input_type=input_type)
        return self._numerator(g, g, g, g)

    # @staticmethod
    # def equal_pair_shapes(n_k):
    #     """Reduced shape family: (i, i, j, j) for i <= j.

    #     NOT the full configuration space -- see even_shapes.  Kept as the default because
    #     it is the family the disconnected null test wants: for a Gaussian field the
    #     (12)(34) Wick pairing needs q1==q2 and q3==q4, so (i,i,j,j) is exactly where the
    #     disconnected piece is maximal.

    #     It is also the only family that fits a covariance estimate: n_k=7 gives 28
    #     shapes -> 28*n_diag = 196 cells, against even_shapes' 406 -> 2842.  Inverting
    #     a sample covariance needs N_sims > p + 2.
    #     """
    #     return [(i, i, j, j) for i in range(n_k) for j in range(i, n_k)]

    @staticmethod
    def even_shapes(n_k):
        """Full parity-even family: (i,j,k,l) with q1<=q2, q3<=q4, (q1,q2)<=(q3,q4).

        The three constraints are exactly the three symmetries of the numerator
        Q[s,B] = sum_x diag_B(g_i g_j)(x) (g_k g_l)(x):
          i<->j      _pair_field is symmetric
          k<->l      same
          (ij)<->(kl)  diag_B is a real even filter, hence self-adjoint:
                       sum_x diag_B(A) B = sum_x A diag_B(B)
        """
        out = []
        for i in range(n_k):
            for j in range(i, n_k):            # q1 <= q2
                for k in range(i, n_k):        # q1 <= q3
                    for l in range(k, n_k):    # q3 <= q4
                        if k == i and l < j:   # q1 == q3  =>  q2 <= q4
                            continue
                        out.append((i, j, k, l))
        return out

    # ------------------------------------------------------------------
    #ideal normalization (mode counting)
    # ------------------------------------------------------------------
    def _unit_shell_maps(self):
        """Shell maps of a unit-amplitude field (delta(k)=1 for all k).
        Used to count mode-quadruplets per config. Returns list of n_k real maps.
        """

        ones_k = self.base.complex_zeros() + 1.0      # delta(k)=1 everywhere
        o_b_list = []
        for b in range(self.n_k):
            o_b = self.base.to_real(self.base.map_utils.fourier_filter(ones_k, 0, self.k_bins[b], self.k_bins[b+1]))
            o_b_list.append(o_b)
        return o_b_list

    def _mode_counts(self, shapes = None):
        """N_modes[s, B]: the quartic estimator applied to unit shell maps.
        ==> number of (k1, k2, k3, k4) with sum zero 
        """
        if shapes is not None:
            o = self._unit_shell_maps()
            return self._numerator(o, o, o, o, shapes=shapes)
        if getattr(self, '_mc_cache', None) is None:
            o = self._unit_shell_maps()
            self._mc_cache = self._numerator(o, o, o, o)
        return self._mc_cache

    def populated(self, N_modes):
        """Cells containing at least one mode quadruplet, i.e. the triangle condition.


        Scalar mode counts only.  mode_counts_odd is sum G^2, not a count, so this does not
        apply there -- and the scalar count being > 0 is necessary but not sufficient for
        N_odd > 0, since an all-coplanar cell has quadruplets but zero weight.
        """
        return np.asarray(N_modes)*float(self.base.gridsize.prod())**3 > 0.5

    def Tk_ideal(self, delta, input_type='real'):
        """Binned trispectrum in continuum units (ideal estimator).

        Returns NaN on cells with no mode quadruplets.
        """
        t = self.Tk_numerator(delta, input_type=input_type)
        N = self._mode_counts()
        good = self.populated(N)
        return np.where(good, t/np.where(good, N, 1.0)*self._discrete_to_continuum(), np.nan)

    # ------------------------------------------------------------------
    # disconnected subtraction (random-map, 4-2+0)
    # ------------------------------------------------------------------
    def generate_sims(self, N_pairs, Pk_input=[], seed0=1000):
        """Generate N_pairs of independent Gaussian random shell-map sets (a, b),
        drawn from power spectrum Pk_input (list [k, P0]; [] = base fiducial).
        Stores self.sims = [(ga_maps, gb_maps), ...]. Match Pk_input to the data
        power for an unbiased subtraction.
        """

        self.sims = []
        self._t0_cache = None      # new maps => new t0; see _t0()
        for n in range(N_pairs):
            fa = self.base.generate_data(seed=seed0 + 2*n, Pk_input=Pk_input, output_type='real')
            fb = self.base.generate_data(seed=seed0 + 2*n + 1, Pk_input=Pk_input, output_type='real')
            self.sims.append((self._process_sim(fa), self._process_sim(fb)))
        return self.sims

    def _t0(self):
        """< Q(a,a,b,b) + Q(a,b,a,b) + Q(a,b,b,a) >_sims / n  -- the random-map-only
        leg of the disconnected subtraction.

        Cached, because it contains no data field: for a fixed self.sims it is the
        same array for every delta.  It is 3*N_pairs of the 1 + 9*N_pairs numerator
        calls per estimate (30 of 91 at N_pairs=10), and it was being recomputed
        identically for every sim in a run.

        The cache is invalidated in generate_sims -- calling that again with a
        different N_pairs / Pk_input / seed0 gives new maps and hence a different
        t0, and a stale value would silently subtract the wrong disconnected term
        rather than raise.
        """
        if getattr(self, '_t0_cache', None) is None:
            Q = self._numerator
            t0 = np.zeros((len(self.shapes), self.n_diag))
            for ga, gb in self.sims:
                t0 += Q(ga, ga, gb, gb) + Q(ga, gb, ga, gb) + Q(ga, gb, gb, ga)
            self._t0_cache = t0 / len(self.sims)
        return self._t0_cache

    def Tk_numerator_connected(self, delta, input_type='real'):
        """Disconnected-subtracted numerator: t4 - t2 + t0.

        With a, b independent GRFs carrying the data's P, <a d> = <a b> = 0, so a Q with
        two random slots survives ONLY via the Wick pairing that pairs those two slots:

            random slots {3,4} or {1,2}  ->  (12)(34) = W1
            random slots {1,3} or {2,4}  ->  (13)(24) = W2
            random slots {1,4} or {2,3}  ->  (14)(23) = W3

        <t4> = T_conn + W1 + W2 + W3, so subtracting an unbiased estimate of W1+W2+W3
        leaves T_conn.  Here

            t2 = < 2 Q(d,d,f,f) + 2 Q(f,d,f,d) + 2 Q(f,d,d,f) >_{f in a,b, sims} / 2n
               -> 2 (W1 + W2 + W3)
            t0 = <   Q(a,a,b,b) +   Q(a,b,a,b) +   Q(a,b,b,a) >_sims / n
               ->     W1 + W2 + W3
            t4 - t2 + t0  ->  T_conn.

        t0 carries no data field, so it comes from the cached _t0(); only t4 and
        t2 are recomputed per delta.
        """
        g = self._process_sim(delta, input_type=input_type)
        Q = self._numerator
        t4 = Q(g, g, g, g)
        t2 = np.zeros_like(t4)
        for ga, gb in self.sims:
            for f in (ga, gb):
                t2 += 2*(Q(g, g, f, f) + Q(f, g, f, g) + Q(f, g, g, f))
        t2 /= (2*len(self.sims))
        return t4 - t2 + self._t0()

    def Tk_ideal_connected(self, delta, input_type='real'):
        """Disconnected-subtracted T in continuum units (ideal).  For the null test.

        Same conventions as Tk_ideal: NaN on empty cells, V^3/N^4 applied.
        """
        t = self.Tk_numerator_connected(delta, input_type=input_type)
        N = self._mode_counts()
        good = self.populated(N)
        return np.where(good, t/np.where(good, N, 1.0)*self._discrete_to_continuum(), np.nan)

    # ------------------------------------------------------------------
    # parity odd part
    # ------------------------------------------------------------------
    def _process_sim_vector(self, delta, input_type = 'real'):
        "Vector shell maps: gv[b][a](x) = IFFT[ i k^a W_b(k) * delta(k) ]. a = x, y, z"
        delta_f = self.applySinv(self._as_input(delta, input_type),
                                 input_type=input_type, output_type='fourier')
        gv = []
       
        k_comps = [
            self.base.k_arrs[0][:,None, None], #k_x
            self.base.k_arrs[1][None, :, None], #k_y
            self.base.k_arrs[2][None, None, :], #k_z
        ]
        for b in range(self.n_k):
            filtered = self.base.map_utils.fourier_filter(delta_f, 0, self.k_bins[b], self.k_bins[b+1])
            gv_b = [self.base.to_real(1j * filtered * k_comp) for k_comp in k_comps]
            gv.append(gv_b)
        return gv

    def _cross_pair_field(self, gv_i, gv_j):
        """Vector pair field C_a(x) = (g_i x g_j)_a(x), a = x, y, z. cross part of the triple product (k1 x k2).k3"""
        
        prod = self.base.map_utils.prod_real
        cross=  [
            prod(gv_i[1], gv_j[2]) - prod(gv_i[2], gv_j[1]), # x
            prod(gv_i[2], gv_j[0]) - prod(gv_i[0], gv_j[2]), # y
            prod(gv_i[0], gv_j[1]) - prod(gv_i[1], gv_j[0]), # z
        ]
        return cross 
    

    def _vector_scalar_pair(self, gv_k, g_l):
        """Vector pair field gv_k[a](x) * g_l(x), a = x, y, z"""
        return [self.base.map_utils.prod_real(gv_k[a], g_l) for a in range(3)]


    @staticmethod
    def odd_shapes(n_k):
        """parity-odd family: (i,j,k,l) with q1<q2, q3<q4, and (i,j) <= (k,l) to remove
        the pair-swap double count.  i==j or k==l vanish identically (antisymmetry)."""
        out = []
        for i in range(n_k):
            for j in range(i+1, n_k):        # q1 < q2
                for k in range(i, n_k):      # q1 <= q3
                    for l in range(k+1, n_k):  # q3 < q4
                        if k == i and l < j:   # q1 == q3  =>  q2 <= q4
                            continue
                        out.append((i, j, k, l))
        return out 

    
    # def _numerator_odd(self, gv, g):
    #     """Odd quartic contraction, shapes_odd x n_diag
    #     Q[s, B] = sum_a sum_x diag_B((gv_i x gv_j)_a)(x) * (gv_k[a]*g_l)(x)"""

    #     out = np.zeros((len(self.shapes_odd), self.n_diag))
    #     for s, (i, j, k, l) in enumerate(self.shapes_odd):
    #         g_ij = self._cross_pair_field(gv[i], gv[j])
    #         g_kl = self._vector_scalar_pair(gv[k], g[l])
    #         for B in range (self.n_diag):
    #             out[s,B] = sum(self.base.map_utils.sum_pair(self._diag_filter(g_ij[a], B), g_kl[a] ) for a in range(3)) 
    #     return out

    def _odd_pair_index(self):
        """Ordered pair list and the (row, col) shape -> pair maps shared by the odd
        numerator and its mode counts."""
        pairs = sorted({(i, j) for i, j, _, _ in self.shapes_odd} |
                       {(k, l) for _, _, k, l in self.shapes_odd})
        pidx = {p: n for n, p in enumerate(pairs)}
        rows = np.array([pidx[(i, j)] for i, j, _, _ in self.shapes_odd])
        cols = np.array([pidx[(k, l)] for _, _, k, l in self.shapes_odd])
        return pairs, rows, cols

    def _pair_contract(self, comps12, comps34, rows, cols, weights=None):
        """out[s,B] = sum_c w_c sum_x diag_B(comps12[rows[s]][c])(x) * comps34[cols[s]][c](x)
        """
        n12, n34, nc = len(comps12), len(comps34), len(comps12[0])
        w = np.ones(nc) if weights is None else np.asarray(weights, dtype=float)

        Cf = []
        for cp in comps12:
            Cf.append([self.base.to_fourier(c) for c in cp])
            del cp[:]                       
        M34 = np.stack([np.concatenate([w[c]*np.asarray(comps34[p][c]).ravel()
                                        for c in range(nc)]) for p in range(n34)])

        out = np.empty((len(rows), self.n_diag))
        G = np.empty((n12, n34))
        for B in range(self.n_diag):
            lo, hi = self.diag_bins[B], self.diag_bins[B+1]
            for c0 in range(0, n12, self.pair_chunk):
                blk = np.stack([
                    np.concatenate([np.asarray(self.base.to_real(
                        self.base.map_utils.fourier_filter(Cf[p][c], 0, lo, hi))).ravel()
                        for c in range(nc)])
                    for p in range(c0, min(c0 + self.pair_chunk, n12))])
                G[c0:c0+len(blk)] = blk @ M34.T
            out[:, B] = G[rows, cols]
        return out

    def _numerator_odd(self, gv, g):
        """
        odd quartic contraction
        Q[s, B] = sum_a sum_x diag_B((gv_i x gv_j)_a)(x) * (gv_k[a]*g_l)(x), based on equation A7 in arxiv.2306.11782
        """
        pairs, rows, cols = self._odd_pair_index()
        cross = [self._cross_pair_field(gv[i], gv[j]) for i, j in pairs]
        vs = [self._vector_scalar_pair(gv[k], g[l]) for k, l in pairs]
        return self._pair_contract(cross, vs, rows, cols)
    
    def Tk_odd_numerator(self, delta, input_type='real'):
        """Raw 4-field odd numerator.

        Returns: array (n_shapes_odd, n_diag), t[s,B] = Q(gv,gv,gv,g)[s,B].
        """
        gv = self._process_sim_vector(delta, input_type=input_type)
        g = self._process_sim(delta, input_type=input_type)
        return self._numerator_odd(gv, g)
    

    # Levi-Civita: (a, c, d, sign) with eps_{acd} = sign.
    _EPS = ((0,1,2,+1), (0,2,1,-1), (1,2,0,+1), (1,0,2,-1), (2,0,1,+1), (2,1,0,-1))

    def _tensor_shell_maps(self):
        """Tensor shell maps T_b^{cd}(x) = IFFT[ k^c k^d W_b(k) ], the delta(k)=1 analogue
        of _process_sim_vector.  Full k (not k-hat), matching the i k^a used there.
        Returns T[b][c][d]; symmetric in (c,d), so only 6 maps per bin are built.
        """
        ones_k = self.base.complex_zeros() + 1.0
        k_comps = [
            self.base.k_arrs[0][:, None, None],
            self.base.k_arrs[1][None, :, None],
            self.base.k_arrs[2][None, None, :],
        ]
        T = []
        for b in range(self.n_k):
            T_b = [[None]*3 for _ in range(3)]
            for c in range(3):
                for d in range(c, 3):
                    T_b[c][d] = T_b[d][c] = self.base.to_real(self.base.map_utils.fourier_filter(
                        k_comps[c]*k_comps[d]*ones_k, 0, self.k_bins[b], self.k_bins[b+1]))
            T.append(T_b)
        return T

    def mode_counts_odd(self):
        """Weighted mode counts N_odd[s, B] = sum_bin [k1.(k2 x k3)]^2.


            [k1.(k2 x k3)]^2 = eps_{acd} eps_{bef} (k1^c k1^e)(k2^d k2^f)(k3^a k3^b)

        so legs 1,2 contract into a rank-2 object M^{ab} and legs 3,4 into T[k]^{ab} * o[l],
        mirroring _numerator_odd's _cross_pair_field / _vector_scalar_pair split, and both
        go through _pair_contract.

        """
        T = self._tensor_shell_maps()
        o = self._unit_shell_maps()
        prod = self.base.map_utils.prod_real
        pairs, rows, cols = self._odd_pair_index()

        eps_a = {}
        for (a, c, d, s) in self._EPS:
            eps_a.setdefault(a, []).append((c, d, s))

        IDX = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
        W6 = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]              # off-diagonals counted twice

        M = []                                            # legs 1,2
        for (i, j) in pairs:
            M_p = []
            for (a, b) in IDX:
                acc = None
                for (c, d, s1) in eps_a[a]:
                    for (e, f, s2) in eps_a[b]:
                        term = s1*s2*prod(T[i][c][e], T[j][d][f])
                        acc = term if acc is None else acc + term
                M_p.append(acc)
            M.append(M_p)
        S = [[prod(T[k][a][b], o[l]) for (a, b) in IDX] for (k, l) in pairs]  # legs 3,4

        return self._pair_contract(M, S, rows, cols, weights=W6)

    

    def _discrete_to_continuum(self):
        """Convert a discrete estimator ratio (numerator / mode counts) to continuum units.

        Parity-agnostic: the derivation below is written for the odd case but the factor is
        the same V^3/N^4 for the even estimator, because it comes entirely from the FFT
        convention and the (2pi)^3 delta_D -> V delta_Kronecker replacement, neither of
        which knows about the weight.  Both Tk_ideal and Tk_odd_ideal apply it.

        base.to_fourier is an unnormalized rfftn and base.to_real carries the 1/N, so the
        grid field is delta_k^FFT = (N/V) delta_cont(k)  (same convention as
        base.generate_data, which builds sqrt(P) * N / sqrt(V)).  For a real-space cell sum
        over four shell maps,

            t      = (1/N^3) sum_{sum k = 0} W * prod delta^FFT   = (N/V^3) sum W (triple)^2 tau_-
            N_odd  = same with delta^FFT -> 1                     = (1/N^3) sum W (triple)^2

        using (2pi)^3 delta_D -> V delta_Kronecker, hence t/N_odd = (N^4/V^3) tau_-.
        Cross-check by dimensions: t ~ L^-3 and N_odd ~ L^-6, so t/N_odd ~ L^3 while
        tau_- ~ L^12 --> the gap is exactly V^3.
        """

        return float(self.base.volume)**3 / float(self.base.gridsize.prod())**4

    # kept so existing notebooks that call the old name keep working
    _discrete_to_continuum_odd = _discrete_to_continuum

    def Tk_odd_ideal(self, delta, input_type="real", normalization='tau'):
        """Normalized parity-odd trispectrum.

        normalization:
          'tau'    (default) divide by mode_counts_odd and apply the discrete->continuum
                   factor, so the return value IS tau_-, directly comparable to theory.
          'scalar' legacy behaviour: divide by the scalar mode count.  This is Coulton+23's
                   own choice (A7: "the particular choice is, to an extent, arbitrary"), and
                   does NOT equal tau_- -- do not compare it to a tau_- curve.
        """
        t = self.Tk_odd_numerator(delta, input_type=input_type)

        if normalization == 'scalar':
            N = self._mode_counts(shapes=self.shapes_odd)
            good = N > 1e-6*N.max()
            return np.where(good, t / np.where(good, N, 1.0), 0.0)

        N = self.mode_counts_odd()
        good = N > 1e-6*N.max()
        return np.where(good, t / np.where(good, N, 1.0) * self._discrete_to_continuum_odd(), 0.0)

    @staticmethod
    def make_parity_odd_ic(base, delta_g, g):
        """delta_PV = delta_G + g*(v1 x v2).v3, v_n^a = IFFT[i k^a |k|^-n delta(k)], n =2,1,0"""

        dk = base.to_fourier(delta_g)
        modk = np.where(base.modk_grid == 0, 1.0, base.modk_grid)
        k_comps = [
            base.k_arrs[0][:,None, None], #k_x
            base.k_arrs[1][None, :, None], #k_y
            base.k_arrs[2][None, None, :], #k_z
        ]
        v = [[base.to_real(1j * kc * modk**(-n) *dk) for kc in k_comps] for n in (2,1,0)]
        v1, v2, v3 = v
        prod = base.map_utils.prod_real
        cross=  [
            prod(v1[1], v2[2]) - prod(v1[2], v2[1]), # x
            prod(v1[2], v2[0]) - prod(v1[0], v2[2]), # y
            prod(v1[0], v2[1]) - prod(v1[1], v2[0]), # z
        ]
        fr123 = cross[0]*v3[0] + cross[1]*v3[1] + cross[2]*v3[2]
        return delta_g + g*fr123, fr123  #parity-odd IC, parity-odd piece

