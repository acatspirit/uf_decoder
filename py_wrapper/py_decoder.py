import numpy as np
import ctypes
import scipy
import sys
import os
import ctypes

def _load_decoder_library():
    # 1. Get the directory where py_decoder.py is located
    wrapper_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Construct the path to the build directory (assuming build/ is at project root)
    # The '..' moves up from py_wrapper/ to the project root
    build_dir = os.path.abspath(os.path.join(wrapper_dir, "..", "build"))

    # 3. Choose the correct file extension based on OS
    ext = ".dylib" if sys.platform == "darwin" else ".so"
    lib_path = os.path.join(build_dir, f"libSpeedDecoder{ext}")

    # 4. Attempt to load
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"Could not find library at {lib_path}")
        
    return ctypes.cdll.LoadLibrary(lib_path)


class UFDecoder:
    def __init__(self, h):  # parity-check matrix h can be scipy coo_matrix, csr_matrix, or numpy array
        self.h = h
        self.n_syndr = self.h.shape[0]  # number of parity checks
        self.n_qbt = self.h.shape[1]  # number of data qubits
        if type(h) != np.ndarray and h.getformat() == 'coo':
            cnt = np.zeros(self.n_syndr, dtype=np.uint8)  # count number of qubits per parity check
            cnt_qbt = np.zeros(self.n_qbt, dtype=np.uint8)  # count number of parity checks per qubit
            for i in self.h.row:
                cnt[i] += 1
            for i in self.h.col:
                cnt_qbt[i] += 1
        elif type(h) != np.ndarray and h.getformat() == 'csr':
            cnt = np.zeros(self.n_syndr, dtype=np.uint8)  # count number of qubits per parity check
            cnt_qbt = np.zeros(self.n_qbt, dtype=np.uint8)  # count number of parity checks per qubit
            for row in range(self.h.shape[0]):
                cnt[row] = len(h.getrow(row).indices)
                for c in h.getrow(row).indices:
                    cnt_qbt[c] += 1
        elif type(h) == np.ndarray:
            cnt_qbt = np.sum(h, axis=0, dtype=np.uint8)
            cnt = np.sum(h, axis=1, dtype=np.uint8)
        else:
            print('invalid parity check matrix')
        self.num_nb_max_syndr = cnt.max()  # maximum number of qubits per parity check
        self.num_nb_max_qbt = cnt_qbt.max()  # maximum number of parity checks per qubit
        self.nn_syndr = np.zeros(self.n_syndr * int(self.num_nb_max_syndr), dtype=np.int32)
        self.nn_qbt = np.zeros(self.n_qbt * int(self.num_nb_max_qbt), dtype=np.int32)
        self.len_nb = np.zeros(self.n_syndr + self.n_qbt, dtype=np.uint8)
        self.correction = np.zeros(self.n_qbt, dtype=np.uint8)
        self.h_matrix_to_tanner_graph()
        # self.decode_lib = ctypes.cdll.LoadLibrary('../build/libSpeedDecoder.so')
        self.decode_lib = _load_decoder_library()

    def add_from_h_row_and_col(self, r, c):
        self.nn_syndr[r * int(self.num_nb_max_syndr) + int(self.len_nb[r + self.n_qbt])] = c
        self.nn_qbt[c * int(self.num_nb_max_qbt) + int(self.len_nb[c])] = r + self.n_qbt
        self.len_nb[r + self.n_qbt] += 1
        self.len_nb[c] += 1

    def h_matrix_to_tanner_graph(self):
        if type(self.h) != np.ndarray and self.h.getformat() == 'coo':
            for i in range(len(self.h.col)):
                c = self.h.col[i]
                r = self.h.row[i]
                self.add_from_h_row_and_col(r, c)
        elif type(self.h) != np.ndarray and self.h.getformat() == 'csr':
            for r in range(self.h.shape[0]):
                for c in self.h.getrow(r).indices:
                    self.add_from_h_row_and_col(r, c)
        elif type(self.h) == np.ndarray:
            for r in range(self.h.shape[0]):
                for c in range(self.h.shape[1]):
                    if self.h[r, c]:
                        self.add_from_h_row_and_col(r, c)

    def decode(self, a_syndrome, a_erasure):
        self.decode_lib.collect_graph_and_decode(ctypes.c_int(self.n_qbt), ctypes.c_int(self.n_syndr), ctypes.c_uint8(self.num_nb_max_qbt), ctypes.c_uint8(self.num_nb_max_syndr),
                                           ctypes.c_void_p(self.nn_qbt.ctypes.data), ctypes.c_void_p(self.nn_syndr.ctypes.data), ctypes.c_void_p(self.len_nb.ctypes.data),
                                           ctypes.c_void_p(a_syndrome.ctypes.data), ctypes.c_void_p(a_erasure.ctypes.data), ctypes.c_void_p(self.correction.ctypes.data))

    def decode_batch(self, a_syndrome, a_erasure, nrep):
        self.decode_lib.collect_graph_and_decode_batch(ctypes.c_int(self.n_qbt), ctypes.c_int(self.n_syndr), ctypes.c_uint8(self.num_nb_max_qbt), ctypes.c_uint8(self.num_nb_max_syndr),
                                           ctypes.c_void_p(self.nn_qbt.ctypes.data), ctypes.c_void_p(self.nn_syndr.ctypes.data), ctypes.c_void_p(self.len_nb.ctypes.data),
                                           ctypes.c_void_p(a_syndrome.ctypes.data), ctypes.c_void_p(a_erasure.ctypes.data), ctypes.c_void_p(self.correction.ctypes.data), ctypes.c_int(nrep))

    def ldpc_decode(self, a_syndrome, a_erasure):
        cluster_sizes = np.zeros(self.n_qbt + self.n_syndr, dtype=np.uint32)
        cluster_count = np.zeros(1, dtype=np.uint32) # total number of clusters
        qubit_cluster_map = np.zeros(self.n_qbt + self.n_syndr, dtype=np.uint32)
        self.decode_lib.ldpc_collect_graph_and_decode(ctypes.c_int(self.n_qbt), ctypes.c_int(self.n_syndr), ctypes.c_uint8(self.num_nb_max_qbt), ctypes.c_uint8(self.num_nb_max_syndr),
                                           ctypes.c_void_p(self.nn_qbt.ctypes.data), ctypes.c_void_p(self.nn_syndr.ctypes.data), ctypes.c_void_p(self.len_nb.ctypes.data),
                                           ctypes.c_void_p(a_syndrome.ctypes.data), ctypes.c_void_p(a_erasure.ctypes.data), ctypes.c_void_p(self.correction.ctypes.data),
                                           ctypes.c_void_p(cluster_sizes.ctypes.data), ctypes.c_void_p(cluster_count.ctypes.data),
                                           ctypes.c_void_p(qubit_cluster_map.ctypes.data))
        final_count = cluster_count[0] 
        final_sizes = cluster_sizes[:final_count].tolist() # get all the real clusters

        cluster_to_qubit_map = {}
        for cluster_index in range(final_count):
            qubit_indices = np.where(qubit_cluster_map == cluster_index)[0]
            cluster_to_qubit_map[cluster_index] = qubit_indices.tolist()
        return final_sizes, cluster_to_qubit_map

    def ldpc_decode_batch(self, a_syndrome, a_erasure, nrep):
        cluster_sizes = np.zeros(self.n_qbt, dtype=np.uint32)
        cluster_count = np.zeros(1, dtype=np.uint32)
        self.decode_lib.ldpc_collect_graph_and_decode_batch(ctypes.c_int(self.n_qbt), ctypes.c_int(self.n_syndr), ctypes.c_uint8(self.num_nb_max_qbt), ctypes.c_uint8(self.num_nb_max_syndr),
                                           ctypes.c_void_p(self.nn_qbt.ctypes.data), ctypes.c_void_p(self.nn_syndr.ctypes.data), ctypes.c_void_p(self.len_nb.ctypes.data),
                                           ctypes.c_void_p(a_syndrome.ctypes.data), ctypes.c_void_p(a_erasure.ctypes.data), ctypes.c_void_p(self.correction.ctypes.data), ctypes.c_int(nrep))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    from some_codes import toric_code, plt_2d_square_toric_code, get_bb

    ########### 2d surface code: ###########
    # build H-matrix/Tanner graph
    L = 40
    H, _ = toric_code(L)
    g = UFDecoder(H)

    # apply noise and get syndromes
    p_err = 0.05
    p_erasure = 0.10
    erasure = np.random.binomial(1, p_erasure, g.n_qbt).astype(np.uint8)
    pauli_err = np.random.binomial(1, p_err, g.n_qbt).astype(np.uint8)
    error = np.logical_or(np.logical_and(np.logical_not(erasure), pauli_err), np.logical_and(erasure, np.random.binomial(1, 0.5, g.n_qbt))).astype(np.uint8)
    syndrome = (g.h @ error % 2).astype(np.uint8)
    plt_2d_square_toric_code(L, error, g.correction, syndrome, g.n_syndr)

    # decode
    t0 = time.perf_counter()
    g.decode(syndrome, erasure)
    t1 = time.perf_counter()
    print('time (s): ', t1 - t0)

    # recompute syndrome to check decoding
    err_plus_correction = np.logical_xor(error, g.correction)
    syndrome = g.h @ err_plus_correction % 2
    print('sum syndrome: ', np.sum(syndrome))
    plt_2d_square_toric_code(L, error, g.correction, syndrome, g.n_syndr)

    ########### bivariate bicycle LDPC code: ###########
    # build H-matrix/Tanner graph
    H, _ = get_bb(6, 6, [3, 1, 2], [3, 1, 2])
    g = UFDecoder(H)

    # apply noise and get syndromes
    for p_err in [0.01*i for i in range(10)]:
        for p_erasure in [0.05*i for i in range(10)]:
            erasure = np.random.binomial(1, p_erasure, g.n_qbt).astype(np.uint8)
            pauli_err = np.random.binomial(1, p_err, g.n_qbt).astype(np.uint8)
            error = np.logical_or(np.logical_and(np.logical_not(erasure), pauli_err), np.logical_and(erasure, np.random.binomial(1, 0.5, g.n_qbt))).astype(np.uint8)
            syndrome = (g.h @ error % 2).astype(np.uint8)

            # decode
            t0 = time.perf_counter()
            g.ldpc_decode(syndrome, erasure)
            t1 = time.perf_counter()
            print('time (s): ', t1 - t0)

            # recompute syndrome to check decoding
            err_plus_correction = np.logical_xor(error, g.correction)
            syndrome = g.h @ err_plus_correction % 2
            print('sum syndrome: ', np.sum(syndrome))

