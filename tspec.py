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

    def __init__(self, base, k_bins, diag_bins, shapes, applySinv=None):
        self.base = base
        self.k_bins = np.asarray(k_bins)
        self.diag_bins = np.asarray(diag_bins)
        self.shapes = list(shapes)
        self.applySinv = applySinv if applySinv is not None else base.applySinv_trivial

        self.n_k = len(self.k_bins) - 1
        self.n_diag = len(self.diag_bins) - 1
        self.shapes_odd = self.odd_shapes(self.n_k)

    # ------------------------------------------------------------------
    # band fitter
    # ------------------------------------------------------------------
    def _process_sim(self, delta, input_type='real'):
        """Return shell maps g_b(x) for every side bin b.

         g_b(x) = IFFT[ W_b(k) * delta(k) ],  W_b = top-hat shell.

        Returns: list of n_k real-space arrays.
        """
        delta_f = self.applySinv(delta, input_type=input_type, output_type='fourier')
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

    def _numerator(self, fw, fx, fy, fz):
        """General quartic contraction Q over four shell-map-lists.

        Q[s=(i,j,k,l), B] = sum_x diag_B(fw_i * fx_j)(x) * (fy_k * fz_l)(x).

        Each of fw, fx, fy, fz is a list of n_k real-space shell maps.
        With all four = the data shell maps, this is the raw 4-field numerator.
        """

        out = np.zeros((len(self.shapes), self.n_diag))
        for s, (i, j, k, l) in enumerate(self.shapes):
            P_ij = self._pair_field(fw[i], fx[j])
            P_kl = self._pair_field(fy[k], fz[l])
            for B in range (self.n_diag):
                out[s,B] = self.base.map_utils.sum_pair(self._diag_filter(P_ij, B), P_kl)
        return out

    def Tk_numerator(self, delta, input_type='real'):
        """Raw 4-field quartic numerator (NO disconnected subtraction).

        Returns: array (n_shapes, n_diag), t[s,B] = Q(g,g,g,g)[s,B].
        """
        g = self._process_sim(delta, input_type=input_type)
        return self._numerator(g, g, g, g)

    @staticmethod
    def equal_pair_shapes(n_k):
        """Reduced shape family: (i, i, j, j) for i <= j."""
        return [(i, i, j, j) for i in range(n_k) for j in range(i, n_k)]

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
        Cached after first call.
        """
        shapes = shapes if shapes is not None else self.shapes
        o = self._unit_shell_maps()
        N = np.zeros((len(shapes), self.n_diag))
        for s, (i,j,k,l) in enumerate(shapes):
            O_ij = self._pair_field(o[i], o[j])
            O_kl = self._pair_field(o[k], o[l])
            for B in range(self.n_diag):
                PB_ij = self._diag_filter(O_ij, B)
                N[s, B] = self.base.map_utils.sum_pair(PB_ij, O_kl)
        return N

    def Tk_ideal(self, delta, input_type='real'):
        """Mode-count normalized trispectrum (ideal estimator). t / N_modes."""

        t = self.Tk_numerator(delta, input_type=input_type)
        N = self._mode_counts()

        good = N > 1e-6*N.max()
        return np.where(good, t / N, 0.0)

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
        for n in range(N_pairs):
            fa = self.base.generate_data(seed=seed0 + 2*n, Pk_input=Pk_input, output_type='real')
            fb = self.base.generate_data(seed=seed0 + 2*n + 1, Pk_input=Pk_input, output_type='real')
            self.sims.append((self._process_sim(fa), self._process_sim(fb)))
        return self.sims

    def Tk_numerator_connected(self, delta, input_type='real'):
        """Disconnected-subtracted numerator: t4 - t2 + t0 (requires generate_sims first).

        t4 = Q(d,d,d,d)
        t2 = <Q(d,d,a,a) + 4 Q(a,d,a,d) + Q(a,a,d,d)
            + Q(d,d,b,b) + 4 Q(b,d,b,d) + Q(b,b,d,d)>_sims / 2
        t0 = <Q(b,b,a,a) + 4 Q(a,b,a,b) + Q(a,a,b,b)>_sims / 2

        Divisor 2n = number of random fields (2 per sim pair): each field's
        6 placements estimate the full t2 target once; t0's 6 placements
        double-cover the 3 Wick pairings. .
        """
    

        g = self._process_sim(delta, input_type=input_type)
        Q = self._numerator
        t4 = Q(g, g, g,g)
        t2 = np.zeros_like(t4)
        t0 = np.zeros_like(t4)
        for ga, gb in self.sims:
            t2 += Q(g,g, ga, ga) + 4*Q(ga, g, ga, g) + Q(ga, ga, g, g)
            t2 += Q(g,g, gb, gb) + 4*Q(gb, g, gb, g) + Q(gb, gb, g, g)
            t0 += Q(gb, gb, ga, ga) + 4*Q(ga, gb, ga,gb) + Q(ga, ga, gb, gb)
        n = len(self.sims)
        t2 /= (2*n)
        t0 /= (2*n)
        return t4 - t2 + t0

    def Tk_ideal_connected(self, delta, input_type='real'):
        """Disconnected-subtracted, mode-count normalized T (ideal). For the null test."""

        t = self.Tk_numerator_connected(delta, input_type=input_type)
        N = self._mode_counts()
        return np.where(N>1e-6*N.max(), t/N, 0.0)

    # ------------------------------------------------------------------
    # parity odd part
    # ------------------------------------------------------------------
    def _process_sim_vector(self, delta, input_type = 'real'):
        "Vector shell maps: gv[b][a](x) = IFFT[ i k^a W_b(k) * delta(k) ]. a = x, y, z"
        delta_f = self.applySinv(delta, input_type=input_type, output_type='fourier')
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
        """parity odd shape family: (i, j, i, j) for i < j (q1 < q2, q3 < q4 Eq. A11 in arxiv2306.11782)"""
        return [(i, j, i, j) for i in range(n_k) for j in range(i+1, n_k)] 
    
    def _numerator_odd(self, gv, g):
        """Odd quartic contraction, shapes_odd x n_diag
        Q[s, B] = sum_a sum_x diag_B((gv_i x gv_j)_a)(x) * (gv_k[a]*g_l)(x)"""

        out = np.zeros((len(self.shapes_odd), self.n_diag))
        for s, (i, j, k, l) in enumerate(self.shapes_odd):
            g_ij = self._cross_pair_field(gv[i], gv[j])
            g_kl = self._vector_scalar_pair(gv[k], g[l])
            for B in range (self.n_diag):
                out[s,B] = sum(self.base.map_utils.sum_pair(self._diag_filter(g_ij[a], B), g_kl[a] ) for a in range(3)) 
        return out
    
    def Tk_odd_numerator(self, delta, input_type='real'):
        """Raw 4-field odd numerator.

        Returns: array (n_shapes_odd, n_diag), t[s,B] = Q(gv,gv,gv,g)[s,B].
        """
        gv = self._process_sim_vector(delta, input_type=input_type)
        g = self._process_sim(delta, input_type=input_type)
        return self._numerator_odd(gv, g)
    

    def Tk_odd_ideal(self, delta, input_type = "real"):
        """Mode-count normalized odd trispectrum (ideal estimator). t / N_modes."""

        t = self.Tk_odd_numerator(delta, input_type=input_type)
        N = self._mode_counts(shapes=self.shapes_odd)

        good = N > 1e-6*N.max()
        return np.where(good, t / N, 0.0)
    

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

