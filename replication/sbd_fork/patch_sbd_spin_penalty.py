#!/usr/bin/env python3
"""Local fork of r-ccs-cms/sbd (branch spin-penalty, 2026-09-09): add H -> H + lambda (S^2 - ss) for the tpb CPU paths.
S^2 = sz(sz+1) + n_minor - sum_pq E^a_pq E^b_qp  (projected on the alpha x beta product space).  Modified files carry a notice.
Idempotent: refuses to patch twice."""
import io, re, sys
R = "/home/alain/src/sbd/include/sbd/chemistry"
det = R + "/basic/determinants.h"; qc = R + "/tpb/qcham.h"; mu = R + "/tpb/mult.h"; sd = R + "/tpb/sbdiag.h"
NOTICE = "// MODIFIED (local fork, 2026-09-09): spin penalty lambda*(S^2 - ss) added; see TwoExciteAB / ZeroExciteS2 in basic/determinants.h\n"
def rd(p): return io.open(p, encoding="utf-8").read()
def wr(p, s): io.open(p, "w", encoding="utf-8", newline="\n").write(s)
DET_DONE = "g_spin_penalty" in rd(det)

# ---- determinants.h: globals + two element functions, inserted right after TwoExcite ----------------------------------
s = rd(det)
anchor = "    return ElemT(sgn) * (I2.Value(A,I,B,J)-I2.Value(A,J,B,I));\n  }\n"
assert DET_DONE or s.count(anchor) == 1
block = anchor + r"""
  // ---- spin penalty (local fork, 2026-09-09) --------------------------------------------------------------------
  // H -> H + g_spin_penalty * (S^2 - g_spin_target), with S^2 = sz(sz+1) + n_minor - sum_pq E^a_pq E^b_qp projected on
  // the alpha x beta product space (the same operator pyscf's selected_ci contract_ss applies).  Spin orbital index =
  // 2*p + spin (alpha even, beta odd, cf. DetFromAlphaBeta).
  inline double g_spin_penalty = 0.0;
  inline double g_spin_target  = 0.0;

  // diagonal: ZeroExcite + lambda * (sz(sz+1) + n_minor - n_double - ss)
  template <typename ElemT, typename DetT>
  ElemT ZeroExciteS2(const DetT & det, const size_t bit_length, const size_t L,
                     const ElemT & I0, const oneInt<ElemT> & I1, const twoInt<ElemT> & I2) {
    ElemT e = ZeroExcite(det, bit_length, L, I0, I1, I2);
    if( g_spin_penalty != 0.0 ) {
      int na = 0, nb = 0, nd = 0;
      for(size_t p = 0; p < L; p++) {
        bool oa = getocc(det, bit_length, 2*p), ob = getocc(det, bit_length, 2*p+1);
        na += oa; nb += ob; nd += (oa && ob);
      }
      double sz = 0.5 * std::abs(na - nb);
      e += ElemT(g_spin_penalty * (sz*(sz+1.0) + std::min(na, nb) - nd - g_spin_target));
    }
    return e;
  }

  // alpha-single x beta-single element: TwoExcite(det, cr_alpha, cr_beta, an_alpha, an_beta) plus the S^2 term
  // -lambda on the pattern cr_alpha/2 == an_beta/2 && an_alpha/2 == cr_beta/2 (E^a_pq E^b_qp), added to whichever of
  // the two integral slots pairs same spins, with TwoExcite's own fermionic sign.
  template <typename ElemT, typename DetT>
  ElemT TwoExciteAB(const DetT & det, const size_t bit_length, int & i, int & j, int & a, int & b,
                    const oneInt<ElemT> & I1, const twoInt<ElemT> & I2) {
    double sgn = 1.0;
    int I = std::min(i,j);
    int J = std::max(i,j);
    int A = std::min(a,b);
    int B = std::max(a,b);
    parity(det,bit_length,std::min(I,A),std::max(I,A),sgn);
    parity(det,bit_length,std::min(J,B),std::max(J,B),sgn);
    if( A > J || B < I ) sgn *= -1.0;
    ElemT dir = I2.Value(A,I,B,J);
    ElemT exc = I2.Value(A,J,B,I);
    if( g_spin_penalty != 0.0 && (i/2 == b/2) && (a/2 == j/2) ) {
      if( (A % 2) == (I % 2) ) dir -= ElemT(g_spin_penalty); else exc -= ElemT(g_spin_penalty);
    }
    return ElemT(sgn) * (dir - exc);
  }
"""
if not DET_DONE:
    s = s.replace(anchor, block)
    if "#include <algorithm>" not in s: s = s.replace("#include <vector>", "#include <vector>\n#include <algorithm>\n#include <cmath>", 1)
    wr(det, NOTICE + s if not s.startswith("//") else s.replace("\n", "\n" + NOTICE, 1))

# ---- qcham.h and mult.h: alpha-beta call sites -> TwoExciteAB; diagonal -> ZeroExciteS2 ---------------------------------
pat = re.compile(r"TwoExcite\((DetI,bit_length,\s*helper\[task\]\.SinglesAlphaCrAnSM\[[^\n]*?\]\[2\*j\+0\],\s*helper\[task\]\.SinglesBetaCrAnSM)")
q = rd(qc); assert "TwoExciteAB" not in q, "qcham.h already patched"
q, nq = pat.subn(r"TwoExciteAB(\1", q); assert nq == 1, nq
q, nz = re.subn(r"hii\[k\] = ZeroExcite\(", "hii[k] = ZeroExciteS2(", q); assert nz == 2, nz
wr(qc, q.replace("\n", "\n" + NOTICE, 1))
m = rd(mu); m, nm = pat.subn(r"TwoExciteAB(\1", m); assert nm == 2, nm
wr(mu, m.replace("\n", "\n" + NOTICE, 1))

# ---- sbdiag.h: options ---------------------------------------------------------------------------------------------------
d = rd(sd)
d, n1 = re.subn(r"(      double eps = 1\.0e-4;\n)", r"\1      double spin_penalty = 0.0;   // local fork: H + spin_penalty*(S^2 - spin_target)\n      double spin_target = 0.0;\n", d); assert n1 == 1
opt = '''	if( std::string(argv[i]) == "--spin_penalty" ) {
	  sbd_data.spin_penalty = std::atof(argv[++i]);
	  sbd::g_spin_penalty = sbd_data.spin_penalty;
	}
	if( std::string(argv[i]) == "--spin_target" ) {
	  sbd_data.spin_target = std::atof(argv[++i]);
	  sbd::g_spin_target = sbd_data.spin_target;
	}
	if( std::string(argv[i]) == "--tolerance" ) {'''
d, n2 = re.subn(r'\tif\( std::string\(argv\[i\]\) == "--tolerance" \) \{', opt.replace("\\", "\\\\"), d, count=1); assert n2 == 1
wr(sd, d.replace("\n", "\n" + NOTICE, 1))
print("patched: determinants.h (+ZeroExciteS2, TwoExciteAB, globals), qcham.h (1 AB site, 2 diagonals), mult.h (2 AB sites), sbdiag.h (--spin_penalty, --spin_target)")
