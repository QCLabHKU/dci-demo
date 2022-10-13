#!/usr/bin/env python

from pyscf import gto, scf, mcscf, ao2mo, fci
from pyscf.fci import cistring as cs
import numpy
from multiprocessing import Pool
from multiprocessing.managers import BaseManager
from functools import reduce
import sys

'''DCI solver'''

def find_occ(string0):
    return numpy.where((string0 & numpy.asarray(
        [1 << i for i in range(int(string0).bit_length())])) != 0)[0].tolist()


def bit_count(int_type):
    count = 0
    while int_type:
        int_type &= int_type - 1
        count += 1
    return count


def sign(i, j, string0):
    if i > j:
        mask = (1 << i) - (1 << (j + 1))
    else:
        mask = (1 << j) - (1 << (i + 1))

    if bit_count(string0 & mask) % 2:
        return -1
    else:
        return 1


def ele_diag(occa, occb, h1e, j, k):

    if type(occa) == int:
        occa = find_occ(occa)
        occb = find_occ(occb)

    ele_diag_term = numpy.einsum('ii->', h1e[occa][:, occa]) + numpy.einsum('ii->', h1e[occb][:, occb]) + (
            numpy.einsum('ij->', j[occa][:, occa]) -
            numpy.einsum('ij->', k[occa][:, occa]) +
            numpy.einsum('ij->', j[occb][:, occb]) -
            numpy.einsum('ij->', k[occb][:, occb])) * 0.5 + numpy.einsum('ij->', j[occa][:, occb])

    return ele_diag_term


def ele_offdiag(stra0, strb0, stra1, strb1, h1e, eri):

    da = stra0 ^ stra1
    db = strb0 ^ strb1

    n1da = bit_count(da)
    n1db = bit_count(db)

    if n1da == 0 and n1db == 2:
        k0 = find_occ(stra0)
        k1 = find_occ(strb0)

        pi = int(db & strb0).bit_length() - 1
        pj = int(db & strb1).bit_length() - 1
        tmp = h1e[pi, pj]

        tmp += numpy.einsum('kk->', eri[:, :, pi, pj][k0][:, k0]) + numpy.einsum(
            'kk->', eri[pi, pj, :, :][k1][:, k1]
        ) - numpy.einsum('kk->', eri[pi, :, :, pj][k1][:, k1])

        return sign(pi, pj, strb1) * tmp

    elif n1da == 2 and n1db == 0:
        k0 = find_occ(strb0)
        k1 = find_occ(stra0)

        pi = int(da & stra0).bit_length() - 1
        pj = int(da & stra1).bit_length() - 1
        tmp = h1e[pi, pj]

        tmp += numpy.einsum('kk->', eri[pi, pj, :, :][k0][:, k0]) + numpy.einsum(
            'kk->', eri[pi, pj, :, :][k1][:, k1]
        ) - numpy.einsum('kk->', eri[pi, :, :, pj][k1][:, k1])

        return sign(pi, pj, stra1) * tmp

    elif n1da == 2 and n1db == 2:
        pi = int(da & stra0).bit_length() - 1
        pj = int(da & stra1).bit_length() - 1
        pk = int(db & strb0).bit_length() - 1
        pl = int(db & strb1).bit_length() - 1

        return sign(pi, pj, stra1) * sign(pk, pl, strb1) * eri[pi, pj, pk, pl]

    elif n1da == 0 and n1db == 4:
        pi = int(db & strb0).bit_length() - 1
        pj = int(db & strb1).bit_length() - 1
        pk = int((db & strb0) ^ (1 << pi)).bit_length() - 1
        pl = int((db & strb1) ^ (1 << pj)).bit_length() - 1
        str1 = strb1 ^ (1 << pi) ^ (1 << pj)

        return sign(pi, pj, strb1) * sign(pk, pl, str1) * (eri[pi, pj, pk, pl] - eri[pi, pl, pk, pj])

    elif n1da == 4 and n1db == 0:

        pi = int(da & stra0).bit_length() - 1
        pj = int(da & stra1).bit_length() - 1
        pk = int((da & stra0) ^ (1 << pi)).bit_length() - 1
        pl = int((da & stra1) ^ (1 << pj)).bit_length() - 1
        str1 = stra1 ^ (1 << pi) ^ (1 << pj)

        return sign(pi, pj, stra1) * sign(pk, pl, str1) * (eri[pi, pj, pk, pl] - eri[pi, pl, pk, pj])

    else:
        return 0


def swap_bits(n, p1, p2):

    bit1 = (n >> p1) & 1
    bit2 = (n >> p2) & 1

    x = (bit1 ^ bit2)
    x = (x << p1) | (x << p2)

    return n ^ x



def initialize(mol, nelec, ncas, ini_nelec, ini_ncas, root, hf='rhf', ini_mo=None,
               multip=1, spin_thres=None, frag_ini=None):
    if type(ini_mo) is not None:
        pass
    if hf == 'rhf':
        mf = scf.RHF(mol)
    elif hf == 'rohf':
        mf = scf.ROHF(mol)
    else:
        raise NotImplementedError

    mf.kernel(verbose=0)
    cas = mcscf.CASCI(mf, ncas, nelec)
    h1e, ecore = cas.get_h1cas(mo_coeff=ini_mo)
    eri = cas.get_h2cas(mo_coeff=ini_mo)
    tmp_eri = ao2mo.restore(1, eri, ncas)
    jdiag = numpy.asarray(numpy.einsum('iijj->ij', tmp_eri), order='C')
    kdiag = numpy.asarray(numpy.einsum('ijji->ij', tmp_eri), order='C')

    ini_cas = mcscf.CASCI(mf, ini_ncas, ini_nelec)
    ini_h1e = ini_cas.get_h1cas(mo_coeff=ini_mo)[0]
    ini_eri = ini_cas.get_h2cas(mo_coeff=ini_mo)

    e, c = initial_solver(name='casci', multip=multip, root=root, h1e=ini_h1e,
                          eri=ini_eri, ncas=ini_ncas, nelec=ini_nelec)

    spin_ini = [fci.spin_op.spin_square(c[i], ini_ncas, ini_nelec) for i in range(len(c))]

    compress_result = compress_cas(e, c, nelec=ini_nelec, nao=ini_ncas, h1e=ini_h1e, eri=ini_eri, multip=multip,
                                   spin_thre=spin_thres, frag_ini=frag_ini)

    return [h1e, tmp_eri, jdiag, kdiag], [e, c, ini_ncas, ini_nelec, spin_ini], compress_result


def initial_solver(name='casci', multip=1, root=0, h1e=None, eri=None, ncas=None, nelec=None):

    if h1e is None or eri is None or ncas is None or nelec is None:
        raise ValueError('Please give required input')

    ss = ((multip - 1) / 2 + 0.5) ** 2 - 0.25
    fs = fci.direct_spin1.FCI()
    fs.conv_tol = 1e-14
    fs = fci.addons.fix_spin_(fs, ss=ss)
    fs.nroots = root + 2
    e, c = fs.kernel(h1e, eri, ncas, nelec)

    return e, c


def comb(x, y):
    return int(numpy.math.factorial(x) / (numpy.math.factorial(x - y) * numpy.math.factorial(y)))


def compress_cas(e, civec, h1e, eri, nao, nelec, frag_ini=None, multip=1, spin_thre=None):
    ss = ((multip - 1) / 2 + 0.5) ** 2 - 0.25
    if spin_thre is None:
        spin_thre = 0.2
    if frag_ini is None:
        frag_ini = 10

    dege = False
    state_of_inter = e[-2]
    eri = ao2mo.restore(1, eri, nao)
    jdiag = numpy.asarray(numpy.einsum('iijj->ij', eri), order='C')
    kdiag = numpy.asarray(numpy.einsum('ijji->ij', eri), order='C')

    if numpy.allclose(e[-1], e[-2]):
        dege = True
        civec = civec[-2:]
    elif len(e) == 2:
        civec = civec[-2]
    elif numpy.allclose(e[-2], e[-3]):
        dege = True
        civec = civec[-3:-1]
    else:
        civec = civec[-2]

    h00, ice, icv, position, temp_sorting, temp_sorting2, trace_spin, orgin_det, ini_projection = \
        None, None, None, None, None, None, None, None, None

    for shape in range(2, 10000):

        rem_l = []
        if dege:
            civec = numpy.array(civec)
            lll = numpy.flip(numpy.argsort(abs(civec.reshape(-1)))[-1 * shape:], axis=0).tolist()
            cishape = civec[0].shape[0] * civec[0].shape[1]
            for j in lll:
                pp = j % cishape
                if pp not in rem_l:
                    rem_l.append(pp)

            ppp = len(lll)
            ooooo = civec[0].reshape(-1)[rem_l]

        else:

            lll = numpy.flip(numpy.argsort(abs(civec.reshape(-1)))[-1 * shape:], axis=0).tolist()
            for j in lll:
                pp = j
                rem_l.append(pp)
            ppp = len(lll)
            ooooo = civec.reshape(-1)[rem_l]

        rem_l = numpy.array(rem_l)
        stra = cs.addrs2str(nao, nelec[0], divmod(rem_l, comb(nao, nelec[1]))[0])
        strb = cs.addrs2str(nao, nelec[1], divmod(rem_l, comb(nao, nelec[1]))[1])

        orgin_det = [Det((stra[i], strb[i])) for i in range(len(stra))]
        shape = rem_l.shape[0]
        com_ih = numpy.zeros((shape, shape))
        for i in range(shape):
            com_ih[i, i] = ele_diag(int(stra[i]), int(strb[i]), h1e, jdiag, kdiag)
            for j in range(i):
                com_ih[i, j] = ele_offdiag(int(stra[i]), int(strb[i]), int(stra[j]), int(strb[j]), h1e, eri)

        h00 = com_ih
        h00 = h00 + h00.T - numpy.diag(h00.diagonal())
        ini_projection = ooooo / numpy.linalg.norm(ooooo)
        ice, icv = numpy.linalg.eigh(h00)

        temp_sorting = [abs(icv[:ppp][:, i] / numpy.linalg.norm(icv[:ppp][:, i]) @ ini_projection)
                        for i in range(h00.shape[0])]
        temp_sorting2 = numpy.array(temp_sorting)[abs(numpy.array(temp_sorting)).argsort()][::-1]
        position = [temp_sorting.index(temp_sorting2[0]), temp_sorting.index(temp_sorting2[1])]
        trace_spin_a = numpy.zeros(civec.shape[-2:])
        trace_spin_b = numpy.zeros(civec.shape[-2:])

        for i in range(len(orgin_det)):
            trace_spin_a[fci.cistring.str2addr(nao, nelec[0], orgin_det[i].a),
                         fci.cistring.str2addr(nao, nelec[1], orgin_det[i].b)] = icv[position[0]][i]
            trace_spin_b[fci.cistring.str2addr(nao, nelec[0], orgin_det[i].a),
                         fci.cistring.str2addr(nao, nelec[1], orgin_det[i].b)] = icv[position[1]][i]

        trace_spin = [fci.spin_op.spin_square(trace_spin_a, nao, nelec),
                      fci.spin_op.spin_square(trace_spin_b, nao, nelec)]
        if shape >= frag_ini:
            if dege:
                if abs(ice[position[0]] - ice[position[1]]) < 1e-6 and abs(trace_spin[0][0] - ss) < spin_thre and \
                        abs(trace_spin[1][0] - ss) < spin_thre:
                    break
            else:
                if abs(trace_spin[0][0] - ss) < spin_thre:
                    break

    assert temp_sorting2[0] * temp_sorting2[0] + temp_sorting2[1] * temp_sorting2[1] > 1 - 1e-2

    return [state_of_inter, h00, ice, icv, trace_spin, position, temp_sorting, orgin_det, dege]



def atd(addr):
    return Det([fci.cistring.addr2str(20, 2, addr // 190), fci.cistring.addr2str(20, 2, addr % 190)])


def screen(inter, init, theta, singlet=False, sc=True):
    large_space = [Det([init.a, init.b], hat=i) for i in sd_generator([2, 2], 20, singlet)]
    qa = []
    if sc:
        for i in large_space:
            if abs(ele_offdiag(init.a, init.b, i.a, i.b, inter.h1e, inter.eri)) > theta:
                qa.append(i)
        return set(qa)
    else:
        return set(large_space)


def sdn(inter, pn, ppp, theta,singlet=False, sc=True):
    tmp = set()
    for j in pn:
        tmp = tmp.union(screen(inter, j, theta, singlet=singlet, sc=sc))
    return tmp - set(ppp) - set(pn)


def diag_rev1(z):
    return diag_rev(z[0], z[1], z[2], z[3], z[4], z[5], z[6])


def diag_rev(inter, sd, t, ppp, pa, pb, eee):

    kk = list(set(sd))
    kkkk = list((set(t)) - (set(kk) | set(ppp)))
    kk_matrix = get_block(inter, kk, diag=True)
    res = numpy.linalg.inv(numpy.eye(len(kk)) * eee - kk_matrix)
    kk_kkkk_matrix = get_block(inter, kk, kkkk)
    kkkk_matrix = get_block(inter, kkkk, diag=True)
    Res = res + res @ kk_kkkk_matrix @ numpy.linalg.inv(numpy.eye(len(kkkk)) * eee - kkkk_matrix - kk_kkkk_matrix.T @
                                                        res @ kk_kkkk_matrix) @ kk_kkkk_matrix.T @ res
    cc = []
    for i in range(len(kk)):
        tmpr = Res[i, i]
        test = sdn(inter, {kk[i]}, set(), theta=1e-6, singlet=False)
        test = list(test - set(kk) - set(ppp) - set(kkkk))
        lab = numpy.array([ele_offdiag(kk[i].a, kk[i].b, j.a, j.b, inter.h1e, inter.eri) for j in test])
        laa = numpy.array([ele_diag(j.a, j.b, inter.h1e, inter.jdiag, inter.kdiag) for j in test])
        cc.append(tmpr + numpy.sum(lab * lab * (1 / (eee - laa - lab * lab * tmpr)).reshape(-1)) * tmpr * tmpr)
    Resolvent = Res.copy()
    for i in range(len(kk)):
        Resolvent[i, i] = cc[i]
    pa_kk_matrix = get_block(inter, pa, kk)
    kk_pb_matrix = get_block(inter, kk, pb)
    ttmp = pa_kk_matrix @ Resolvent @ kk_pb_matrix

    return ttmp



def sd_generator(nelec, nao, singlet=False):
    for i in range(5):
        if i == 0:
            for alpha_occ in range(nelec[0]):
                for alpha_vir in range(nelec[0], nao):
                    yield [[alpha_occ], [alpha_vir]]
        elif i == 1:
            for beta_occ in range(nelec[1]):
                for beta_vir in range(nelec[1], nao):
                    yield [[beta_occ], [-1 * beta_vir]]
            if singlet:
                break

        elif i == 2:
            for alpha_occ_a in range(nelec[0]):
                for alpha_occ_b in range(alpha_occ_a + 1, nelec[0]):
                    for alpha_vir_a in range(nelec[0], nao):
                        for alpha_vir_b in range(alpha_vir_a + 1, nao):
                            yield [[alpha_occ_a, alpha_occ_b], [alpha_vir_a, alpha_vir_b]]

        elif i == 3:
            for beta_occ_a in range(nelec[1]):
                for beta_occ_b in range(beta_occ_a + 1, nelec[1]):
                    for beta_vir_a in range(nelec[1], nao):
                        for beta_vir_b in range(beta_vir_a + 1, nao):
                            yield [[beta_occ_a, beta_occ_b], [beta_vir_a * -1, beta_vir_b * -1]]

        else:
            for alpha_occ in range(nelec[0]):
                for alpha_vir in range(nelec[0], nao):
                    for beta_occ in range(nelec[1]):
                        for beta_vir in range(nelec[1], nao):
                            yield [[alpha_occ, beta_occ], [alpha_vir, -1 * beta_vir]]


def det_to_frag(det):
    def inside_find_occ(string0):
        if string0 == 0:
            return []
        return numpy.where((string0 & numpy.asarray([1 << temp_i_inside for temp_i_inside in
                                                     range(int(string0).bit_length())])) != 0)[0].tolist()

    frag = [[], []]
    len_of_det_a = bit_count(det.a)
    len_of_det_b = bit_count(det.b)
    occ_a = inside_find_occ((2 ** len_of_det_a - 1) ^ det.a)
    occ_b = inside_find_occ((2 ** len_of_det_b - 1) ^ det.b)
    for i in occ_a:
        if i < len_of_det_a:
            frag[0].append(i)
        else:
            frag[1].append(i)
    for i in occ_b:
        if i < len_of_det_b:
            frag[0].append(i)
        else:
            frag[1].append(i * -1)
    return frag


def get_frag_based_q(frag, nelectron, naos):
    tmp_target_q = set()
    for i in frag:
        tmp_target_q = tmp_target_q.union({Det([i.a, i.b], hat=j) for j in sd_generator(nelectron, naos)})
    return list(tmp_target_q - set(frag))


class Det:
    def __init__(self, i, hat=None):
        self.a = int(i[0])
        self.b = int(i[1])

        if hat is not None:
            indexa = find_occ(self.a) + list(set(range(100)) - set(find_occ(self.a) ))
            indexb = find_occ(self.b) + list(set(range(100)) - set(find_occ(self.b) ))
            for j in range(len(hat[0])):
                if hat[1][j] > 0:
                    self.a = swap_bits(self.a, indexa[hat[0][j]], indexa[hat[1][j]])
                else:
                    self.b = swap_bits(self.b, indexb[hat[0][j]], indexb[abs(hat[1][j])])

    def __eq__(self, other):
        return self.a == other.a and self.b == other.b

    def __hash__(self):
        return hash((self.a, self.b))


def get_block(inter, block_a, block_b=None, diag=False):
    inter = inter
    if all([inter.h1e is None, inter.eri is None,
            inter.jdiag is None, inter.kdiag is None]):
        raise ValueError("Please state the integral")
    if diag:
        if block_b:
            raise ValueError("please read the function carefully")

        matrix = numpy.zeros((len(block_a), len(block_a)))

        for i in range(len(block_a)):
            matrix[i, i] = ele_diag(block_a[i].a, block_a[i].b, inter.h1e, inter.jdiag, inter.kdiag)
            for j in range(i+1, len(block_a)):
                matrix[i, j] = ele_offdiag(block_a[i].a, block_a[i].b, block_a[j].a, block_a[j].b, inter.h1e, inter.eri)
        matrix = matrix + matrix.T - numpy.diag(matrix.diagonal())
        return matrix

    else:
        if block_b is None:
            raise ValueError("please read the function carefully")

        matrix = numpy.zeros((len(block_a), len(block_b)))
        for i in range(len(block_a)):
            for j in range(len(block_b)):
                matrix[i, j] = ele_offdiag(block_a[i].a, block_a[i].b, block_b[j].a, block_b[j].b, inter.h1e, inter.eri)
        return matrix


def schur(fragments_l, fragments_r=None, diag=True, nelec=None, nao=None, ene=None, start=500, step=200, final=100000):
    inter = SchurStorage()
    if nelec is None:
        nelec = [bit_count(fragments_l[0].a), bit_count(fragments_l[0].b)]
    if nao is None:
        raise ValueError("Please give the ao number")

    if diag:
        fragments = fragments_r
    else:
        fragments = fragments_r.extend(fragments_l)

    unranked_target_q_ini_part_a = get_frag_based_q(fragments, nelec, nao)
    unranked_fragq_ele_diag = [ele_diag(i.a, i.b, inter.h1e, inter.jdiag,
                                        inter.kdiag) for i in unranked_target_q_ini_part_a]
    unranked_block_frag_q_ini = get_block(fragments, unranked_target_q_ini_part_a, diag=False)
    unranked_ioo_pt = [unranked_block_frag_q_ini[:, i].reshape(len(fragments), 1) @
                       unranked_block_frag_q_ini[:, i].reshape(1, len(fragments)) *
                       (1 / (ene - unranked_fragq_ele_diag[i]))
                       for i in range(len(unranked_target_q_ini_part_a))]

    if diag:
        index_ini = numpy.einsum('ij,ij->j', abs(unranked_block_frag_q_ini),
                                 abs(unranked_block_frag_q_ini)).argsort()[::-1]
    else:
        index_ini = numpy.einsum('ij,kj->j', abs(unranked_block_frag_q_ini[:len(fragments_l)]),
                                 abs(unranked_block_frag_q_ini[len(fragments_r):])).argsort()[::-1]
    lsp = []
    bbb = get_block([unranked_target_q_ini_part_a[i] for i in index_ini[:start]], diag=True)

    bbbinv = numpy.linalg.inv(numpy.eye(start) * ene - bbb)
    q_size, h_eff_npt_revise_wpt = None, None

    def edge_b(edge_b_bbbinv, local_fragments):

        revise_edge_b = 1

        edge_b_unranked_target_q_ini_part_a = set(get_frag_based_q(local_fragments, nelec, nao)) \
                                              - set(unranked_target_q_ini_part_a) - set(fragments)
        if diag:
            phqr = unranked_block_frag_q_ini[:, index_ini[:edge_b_bbbinv.shape[0]]] @ edge_b_bbbinv
        else:
            phqri = unranked_block_frag_q_ini[:len(fragments_r), index_ini[:edge_b_bbbinv.shape[0]]] @ edge_b_bbbinv
            phqrj = unranked_block_frag_q_ini[len(fragments_r):, index_ini[:edge_b_bbbinv.shape[0]]] @ edge_b_bbbinv
        for i in edge_b_unranked_target_q_ini_part_a:

            tmp_coup = 1
            if numpy.linalg.norm(tmp_coup) >= 1e-8:
                coup_const = 1
                revise_edge_b += coup_const * tmp_coup

        return revise_edge_b

    for i in range(start + step, final, step):

        len_old = bbbinv.shape[0]
        tempq = get_block([unranked_target_q_ini_part_a[tmp_i] for tmp_i in index_ini[:len_old]],
                          block_b=[unranked_target_q_ini_part_a[tmp_j] for tmp_j in index_ini[len_old:i]], diag=False)
        temqq = get_block([unranked_target_q_ini_part_a[tmp_i] for tmp_i in index_ini[len_old:i]], diag=True)

        sche = numpy.linalg.inv((numpy.eye(temqq.shape[0]) * ene - temqq) - tempq @ bbbinv @ tempq.T)
        block_b = bbbinv @ tempq.T @ sche
        bbbinv = numpy.hstack((numpy.vstack((bbbinv + bbbinv @ tempq.T @ sche @ tempq @ bbbinv, block_b.T)),
                               numpy.vstack((block_b, sche))))

        if diag:
            h_eff_npt = unranked_block_frag_q_ini[:, index_ini[:i]] @ bbbinv \
                        @ unranked_block_frag_q_ini[:, index_ini[:i]].T

            h_eff_npt_revise = edge_b(h_eff_npt,
                                      [unranked_target_q_ini_part_a[tmp_j] for tmp_j in index_ini[len_old:i]])
        else:
            h_eff_npt = unranked_block_frag_q_ini[:len(fragments_r), index_ini[:i]] @ bbbinv \
                        @ unranked_block_frag_q_ini[len(fragments_r):, index_ini[:i]].T

            h_eff_npt_revise = edge_b(h_eff_npt,
                                      [unranked_target_q_ini_part_a[tmp_j] for tmp_j in index_ini[len_old:i]])

        lsp.append(h_eff_npt_revise)

        if len(lsp) > 4:
            if numpy.linalg.norm(lsp[-1] - lsp[-4]) < 1e-5:
                q_size = i
                h_eff_npt_revise_wpt = h_eff_npt_revise + sum([unranked_ioo_pt[tmp_i] for tmp_i in index_ini[:q_size]])
                break

    if diag:
        psi_b = bbbinv @ unranked_block_frag_q_ini[:, index_ini[:q_size]].T
        index_renew = [index_ini[i] for i in numpy.einsum('ij,ij->i', abs(psi_b), abs(psi_b)).argsort()[::-1]]
        fragment_candidate = [[unranked_target_q_ini_part_a[tmp_i0]
                               for tmp_i0 in [index_ini[i] for i in index_renew]][:len(fragments_l)*2],
                              psi_b[index_renew, :][:len(fragments_l)*2]]
        return h_eff_npt_revise_wpt, fragment_candidate
    else:
        return h_eff_npt_revise_wpt


def post_care(hamiltonian, fragment_can, pre_c, pre_e, ncas, nelec, frag_adding_terms):

    e, c = numpy.linalg.eigh(hamiltonian)

    state_tracing = numpy.argsort(numpy.array([sum(abs(c[:, tmp_ip] * pre_c))
                                               for tmp_ip in range(hamiltonian.shape[0])]))[0]
    pre_e.append(e[state_tracing])

    if len(pre_e) < 5:
        converged = False
    else:
        if (pre_e[-1] - pre_e[-4]) < 1e-5:
            converged = True
        else:
            converged = False
    candidate_indexing = [0]
    candidate_ordering = []

    for i in range(len(fragment_can)):
        for j in fragment_can[i]:
            candidate_indexing.append(j[0])
            candidate_ordering.append(j[1]*c[:, state_tracing][i])

    adding_index = numpy.argsort(abs(numpy.array(candidate_ordering)))[::-1]

    revised_frag_set = None
    for i in range(hamiltonian.shape[0] + frag_adding_terms, len(candidate_indexing)):
        revised_frag_set = {candidate_indexing[tmp_iq] for tmp_iq in adding_index[:i]}
        if len(revised_frag_set) == hamiltonian.shape[0] + frag_adding_terms:
            break


    return pre_e, converged, revised_frag_set


class SchurStorage:
    def __init__(self, h1ee, erii, jdiagg, kdiagg):
        self.h1e, self.eri, self.jdiag, self.kdiag = h1ee, erii, jdiagg, kdiagg


class DCI(object):

    def __init__(self, mol, ini_ncas,
                 nelec=None, ncas=None, ini_nelec=None,
                 multip=1, root=0, hf_method='rhf', ini_mo=None,
                 spin_thres=0.2, frag_conv=1e-5, frag_ini=10, nproc=8, theta=[1e-8,1e-6]):
        if type(mol) != gto.mole.Mole:
            raise TypeError('pyscf.gto.mole.Mole object required!')

        if ini_ncas is None:
            raise KeyError('Please give the initial cas space size (one-body)')
        else:
            self.ini_ncas = ini_ncas

        self.mol = mol
        self.nelec = self.mol.nelec if not nelec else nelec
        self.ncas = self.mol.nao if not ncas else ncas
        self.ini_nelec = self.nelec if not ini_nelec else ini_nelec
        self.root = root
        self.hf_method = hf_method
        self.ini_mo = ini_mo
        self.multip = multip
        self.spin_thres = spin_thres
        self.frag_ini = frag_ini
        self.frag_conv = frag_conv


        self.nproc = nproc
        self.map = Pool(self.nproc).map

        self.fragment = None
        self.converged = False

        tmp_global, tmp_ini, tmp_com = initialize(mol=self.mol, nelec=self.nelec, ncas=self.ncas,
                                                  ini_nelec=self.ini_nelec, ini_ncas=self.ini_ncas,
                                                  root=self.root, hf='rhf', ini_mo=ini_mo,
                                                  multip=self.multip, spin_thres=self.spin_thres,
                                                  frag_ini=self.frag_ini)

        self.inter = SchurStorage(h1ee=tmp_global[0], erii=tmp_global[1], jdiagg=tmp_global[2], kdiagg=tmp_global[3])
        self.e_cas, self.c_cas, self.nelec_cas, self.ncas_cas, self.spin_cas = tmp_ini
        self.soi_com, self.h00_com, self.e_com, self.v_com, self.trace_spin_com, \
            self.position_com, self.temp_sorting_com, self.orgin_det_com, self.dege_com = tmp_com


        self.hamil = self.h00_com

        tmp_diag = self.hamil.diagonal().copy()
        tmp_index_diag = numpy.argsort(tmp_diag)
        tmp_diag.sort()
        tmp_init = True
        self.fragment = []
        for tmp_i in range(len(tmp_diag)):
            if tmp_init:
                self.fragment.append([])
                self.fragment[-1].append(self.orgin_det_com[tmp_index_diag[tmp_i]])
                tmp_init = False
            else:
                if abs(tmp_diag[tmp_i] - tmp_diag[tmp_i-1]) < 1e-2:
                    self.fragment[-1].append(self.orgin_det_com[tmp_index_diag[tmp_i]])
                else:
                    self.fragment.append([])
                    self.fragment[-1].append(self.orgin_det_com[tmp_index_diag[tmp_i]])

        self.e = [self.soi_com]

    def post_care(self):

        self.e, self.converged, self.fragment = 0, 0, 0

        return 0

    def hamil_cons(self):

        frag = reduce(lambda x, y: x + y, self.fragment)
        ind_tmp = [len(i) for i in self.fragment]
        ham_index = [sum(ind_tmp[:i]) for i in range(len(self.fragment))]
        ham_index.append(len(frag))

        sds = [sdn(self.inter, self.fragment[i], frag, theta=1e-8) for i in range(len(self.fragment))]
        ts = [list(sdn(self.inter, sdi, frag, theta=1e-6, singlet=True)) for sdi in sds]

        f0, f3, f6 = [self.inter] * len(frag), [frag] * len(frag), [self.e] * 6
        f1, f2, f4, f5 = [], [], [], []
        index_a, index_b = [], []

        for i in range(len(sds)):
            for j in range(i, len(sds)):
                f4.append(self.fragment[i])
                f5.append(self.fragment[j])
                index_a.append(i)
                index_b.append(j)
                if i == j:
                    f1.append(sds[i])
                    f2.append(ts[i])
                else:
                    f1.append(set(sds[i]) | set(sds[j]))
                    f2.append(set(ts[i]) | set(ts[j]))

        inputtmp = zip(f0, f1, f2, f3, f4, f5, f6)

        hamil_shape = len(frag)
        pol = Pool(6)
        tmppp = pol.map(diag_rev1, inputtmp)
        pol.close()

        hamil_revise = numpy.zeros((hamil_shape, hamil_shape))
        lpp = get_block(self.inter, frag, diag=True)
        for i in range(len(tmppp)):
            if index_a[i] == index_b[i]:
                hamil_revise[ham_index[index_a[i]]:ham_index[index_a[i] + 1],
                             ham_index[index_b[i]]:ham_index[index_b[i] + 1]] = tmppp[i]
            else:
                hamil_revise[ham_index[index_a[i]]:ham_index[index_a[i] + 1],
                             ham_index[index_b[i]]:ham_index[index_b[i] + 1]] = tmppp[i]
                hamil_revise[ham_index[index_b[i]]:ham_index[index_b[i] + 1],
                             ham_index[index_a[i]]:ham_index[index_a[i] + 1]] = tmppp[i].T
        self.hamil = hamil_revise + lpp


        return 0

    @classmethod
    def schur(cls, position=None):

        if position[0] == position[1]:
            matrix = numpy.zeros((2, 2))

        else:
            matrix = numpy.zeros((2, 2))
        return matrix, position

if __name__ == '__main__':
    verbose = 0
    a = gto.M(atom='h 0 0 0; h 0 0 1; h 0 1 0; h 0 1 1', basis='cc-pvdz')
    cc2 = DCI(a, ini_ncas=10, nelec=[2, 2], ncas=20, ini_nelec=[2, 2], multip=3, root=0,
              hf_method='rohf', ini_mo=numpy.load('mo.npy'), spin_thres=0.2, frag_conv=1e-5,
              frag_ini=5, nproc=6, theta=[1e-8, 1e-6])
    cycle = 0
    print("cycle 0, electronic energy is {:+.6f}, error is {:+.6f}Ha".format(cc2.e[0], cc2.e[0] - -4.950176))
    while not cc2.converged:
        cycle += 1
        old_e = cc2.e
        cc2.hamil_cons()
        cc2.e = numpy.linalg.eigh(cc2.hamil)[0][0]
        print("cycle {}, electronic energy is {:+.6f}, abs error is {:+.6f}Ha".format(cycle, cc2.e,cc2.e - -4.950176))
        if abs(cc2.e - old_e) < 1e-4:
            cc2.converged = 1



