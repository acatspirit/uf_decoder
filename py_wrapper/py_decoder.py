import os
import sys

import numpy as np
import ctypes
import scipy


class UFDecoder:
    def __init__(self, h):  # parity-check matrix h can be scipy coo_matrix, csr_matrix, or numpy array
        self.h = h
        self.n_syndr = self.h.shape[0]  # number of parity checks
        self.n_qbt = self.h.shape[1]  # number of data qubits
        if type(h) != np.ndarray and h.getformat() == 'coo':
            cnt = np.zeros(self.n_syndr, dtype=np.uint32)  # count number of qubits per parity check
            cnt_qbt = np.zeros(self.n_qbt, dtype=np.uint32)  # count number of parity checks per qubit
            for i in self.h.row:
                cnt[i] += 1
            for i in self.h.col:
                cnt_qbt[i] += 1
        elif type(h) != np.ndarray and h.getformat() == 'csr':
            cnt = np.zeros(self.n_syndr, dtype=np.uint32)  # count number of qubits per parity check
            cnt_qbt = np.zeros(self.n_qbt, dtype=np.uint32)  # count number of parity checks per qubit
            for row in range(self.h.shape[0]):
                cnt[row] = len(h.getrow(row).indices)
                for c in h.getrow(row).indices:
                    cnt_qbt[c] += 1
        elif type(h) == np.ndarray:
            cnt_qbt = np.sum(h, axis=0, dtype=np.uint32)
            cnt = np.sum(h, axis=1, dtype=np.uint32)
        else:
            print('invalid parity check matrix')
        self.num_nb_max_syndr = cnt.max()  # maximum number of qubits per parity check
        self.num_nb_max_qbt = cnt_qbt.max()  # maximum number of parity checks per qubit
        print('max number of qubits per parity check: ', self.num_nb_max_syndr)
        print('max number of parity checks per qubit: ', self.num_nb_max_qbt)
        self.nn_syndr = np.zeros(self.n_syndr * int(self.num_nb_max_syndr), dtype=np.int32)
        self.nn_qbt = np.zeros(self.n_qbt * int(self.num_nb_max_qbt), dtype=np.int32)
        self.len_nb = np.zeros(self.n_syndr + self.n_qbt, dtype=np.uint32)
        self.correction = np.zeros(self.n_qbt, dtype=np.uint32)
        self.h_matrix_to_tanner_graph()


        # changed from the original path to the one in the build folder, since the .so file is generated there
        # self.decode_lib = ctypes.cdll.LoadLibrary('../build/libSpeedDecoder.so')
        # 1. Get the directory where py_decoder.py is located
        _wrapper_dir = os.path.dirname(os.path.abspath(__file__))

        # 2. Point to the build directory (which is now adjacent to py_wrapper/ in the root)
        # We check for both .dylib (macOS) and .so (Linux)
        _lib_name = "libSpeedDecoder.dylib" if sys.platform == "darwin" else "libSpeedDecoder.so"
        # _lib_path = os.path.join(os.path.dirname(_wrapper_dir), "build", _lib_name)
        _lib_path = os.path.abspath(os.path.join(_wrapper_dir, "..", "build", _lib_name))

        # 3. Load it
        self.decode_lib = ctypes.cdll.LoadLibrary(_lib_path)
        

    def add_from_h_row_and_col(self, r, c):
        self.nn_syndr[r * int(self.num_nb_max_syndr) + int(self.len_nb[r + self.n_qbt])] = c
        self.nn_qbt[c * int(self.num_nb_max_qbt) + int(self.len_nb[c])] = r + self.n_qbt
        self.len_nb[r + self.n_qbt] += 1
        self.len_nb[c] += 1
        if self.len_nb[r + self.n_qbt] > self.num_nb_max_syndr:
            print("Error: Too many neighbors for syndrome node")
        if self.len_nb[c] > self.num_nb_max_qbt:
            print("Error: Too many neighbors for qubit node")

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
        # Before calling ldpc_collect_graph_and_decode
        # Before calling the C function
        # print("Max index in nn_qbt:", np.max(self.nn_qbt))
        # print("Min index in nn_qbt:", np.min(self.nn_qbt))
        assert np.all(self.nn_qbt >= 0), "Negative index found in connectivity list!"
        assert np.all(self.nn_qbt < self.n_qbt + self.n_syndr), "Index found larger than graph size!"
        assert np.max(self.nn_qbt) < (self.n_qbt + self.n_syndr), "Graph adjacency list contains invalid indices"
        self.decode_lib.collect_graph_and_decode(ctypes.c_int(self.n_qbt), ctypes.c_int(self.n_syndr), ctypes.c_uint32(self.num_nb_max_qbt), ctypes.c_uint32(self.num_nb_max_syndr),
                                           ctypes.c_void_p(self.nn_qbt.ctypes.data), ctypes.c_void_p(self.nn_syndr.ctypes.data), ctypes.c_void_p(self.len_nb.ctypes.data),
                                           ctypes.c_void_p(a_syndrome.ctypes.data), ctypes.c_void_p(a_erasure.ctypes.data), ctypes.c_void_p(self.correction.ctypes.data))

    def decode_batch(self, a_syndrome, a_erasure, nrep):
        self.decode_lib.collect_graph_and_decode_batch(ctypes.c_int(self.n_qbt), ctypes.c_int(self.n_syndr), ctypes.c_uint8(self.num_nb_max_qbt), ctypes.c_uint8(self.num_nb_max_syndr),
                                           ctypes.c_void_p(self.nn_qbt.ctypes.data), ctypes.c_void_p(self.nn_syndr.ctypes.data), ctypes.c_void_p(self.len_nb.ctypes.data),
                                           ctypes.c_void_p(a_syndrome.ctypes.data), ctypes.c_void_p(a_erasure.ctypes.data), ctypes.c_void_p(self.correction.ctypes.data), ctypes.c_int(nrep))

    def ldpc_decode(self, a_syndrome, a_erasure, max_clusters=1000):
        """
        Decodes a single shot and returns:
          - A list of valid cluster sizes
          - A dictionary mapping cluster index IDs to specific qubit identity lists
        """
        # Pre-allocate reference numpy arrays for memory mapping
        cluster_sizes_out = np.zeros(max_clusters, dtype=np.int32)
        cluster_count_out = np.zeros(1, dtype=np.int32)
        qubit_cluster_map_out = np.zeros(self.n_qbt, dtype=np.int32) # NEW: qubit mapping buffer

        # Call C shared object with the additional parameter (Argument 13)
        self.decode_lib.ldpc_collect_graph_and_decode(
            ctypes.c_int(self.n_qbt), 
            ctypes.c_int(self.n_syndr), 
            ctypes.c_uint8(self.num_nb_max_qbt), 
            ctypes.c_uint8(self.num_nb_max_syndr),
            ctypes.c_void_p(self.nn_qbt.ctypes.data), 
            ctypes.c_void_p(self.nn_syndr.ctypes.data), 
            ctypes.c_void_p(self.len_nb.ctypes.data),
            ctypes.c_void_p(a_syndrome.ctypes.data), 
            ctypes.c_void_p(a_erasure.ctypes.data), 
            ctypes.c_void_p(self.correction.ctypes.data),
            ctypes.c_void_p(cluster_sizes_out.ctypes.data),  
            ctypes.c_void_p(cluster_count_out.ctypes.data),
            ctypes.c_void_p(qubit_cluster_map_out.ctypes.data) # NEW POINTER PASSED DOWN
        )

        actual_count = cluster_count_out[0]
        final_sizes = cluster_sizes_out[:actual_count].tolist()

        # Parse the flat qubit cluster map into structured lists group entries
        cluster_to_qubits_dict = {}
        for c_idx in range(1, actual_count + 1):
            # Locate all data qubit positions belonging to cluster identifier c_idx
            matching_qubit_ids = np.where(qubit_cluster_map_out == c_idx)[0].tolist()
            cluster_to_qubits_dict[c_idx - 1] = matching_qubit_ids

        return final_sizes, cluster_to_qubits_dict

    def ldpc_decode_batch(self, a_syndrome, a_erasure, nrep, max_clusters_per_rep=100):
        """
        Decodes a batch of noise repetitions and returns a list of tuples:
          [(shot_0_sizes, shot_0_membership_dict), (shot_1_sizes, shot_1_membership_dict), ...]
        """
        # Pre-allocate flat contiguous memory spaces for ctypes tracking
        cluster_sizes_out = np.zeros(nrep * max_clusters_per_rep, dtype=np.int32)
        cluster_counts_out = np.zeros(nrep, dtype=np.int32)
        qubit_maps_out = np.zeros(nrep * self.n_qbt, dtype=np.int32) # Matrix map matrix buffer

        # Execute C binary matching the expanded signature parameters (Argument 15)
        self.decode_lib.ldpc_collect_graph_and_decode_batch(
            ctypes.c_int(self.n_qbt), 
            ctypes.c_int(self.n_syndr), 
            ctypes.c_uint8(self.num_nb_max_qbt), 
            ctypes.c_uint8(self.num_nb_max_syndr),
            ctypes.c_void_p(self.nn_qbt.ctypes.data), 
            ctypes.c_void_p(self.nn_syndr.ctypes.data), 
            ctypes.c_void_p(self.len_nb.ctypes.data),
            ctypes.c_void_p(a_syndrome.ctypes.data), 
            ctypes.c_void_p(a_erasure.ctypes.data), 
            ctypes.c_void_p(self.correction.ctypes.data), 
            ctypes.c_int(nrep),
            ctypes.c_void_p(cluster_sizes_out.ctypes.data),        
            ctypes.c_void_p(cluster_counts_out.ctypes.data),       
            ctypes.c_int(max_clusters_per_rep),
            ctypes.c_void_p(qubit_maps_out.ctypes.data) # NEW: Passed matrix handle block pointer
        )

        # Re-pack the flat contiguous output tracking matrices into Python structures
        batch_results = []
        for r in range(nrep):
            count = cluster_counts_out[r]
            
            # Slice the sizes list for this specific repetition block
            size_start = r * max_clusters_per_rep
            shot_sizes = cluster_sizes_out[size_start:size_start + count].tolist()
            
            # Slice the flat map tracking segment matching this current shot's qubit region
            map_start = r * self.n_qbt
            shot_qubit_map = qubit_maps_out[map_start:map_start + self.n_qbt]
            
            # Group specific qubit indices into clean cluster lists matching their IDs
            shot_membership = {}
            for c_id in range(1, count + 1):
                matching_ids = np.where(shot_qubit_map == c_id)[0].tolist()
                shot_membership[c_id - 1] = matching_ids
                
            batch_results.append((shot_sizes, shot_membership))

        return batch_results

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

