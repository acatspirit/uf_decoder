#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <stdint.h>
#include "../inc/global.h"
#include "../inc/graph_type.h"
#include "../inc/stabilizer_main.h"

/* get the next prime number above the next power of two (for hash table) */
static int getNextPrimePower2(int N){
  if(N <= 16) return 17;
  else if(N <= 32) return 37;
  else if(N <= 64) return 67;
  else if(N <= 128) return 131;
  else if(N <= 256) return 257;
  else if(N <= 512) return 521;
  else if(N <= 1024) return 1031;
  else if (N <= 2048) return 2053;
  else if (N <= 4096) return 4099;
  else if (N <= 8192) return 8209;
  else if (N <= 16384) return 16411;
  else if (N <= 32768) return 32771;
  else if (N <= 65536) return 65537;
  else if (N <= 131072) return 131101;
  else if (N <= 262144) return 262147;
  else if (N <= 524288) return 524309;
  else if (N <= 1048576) return 1048583;
  else if (N <= 2097152) return 2097169;
  else if (N <= 4194304) return 4194319;
  else if (N <= 8388608) return 8388617;
  else if (N <= 16777216) return 16777259;
  else if (N <= 33554432) return 33554467;
  else if (N <= 67108864) return 67108879;
  else if (N <= 134217728) return 134217757;
  else if (N <= 268435456) return 268435459;
  else if (N <= 536870912) return 536870923;
  else if (N <= 1073741824) return 1073741827;
  else return 2147483647; // integer maximum
}

static bool isVisited(int idx, int* hashTab, int tabSize){
  int h = idx % tabSize;
  while(hashTab[h] >= 0 && hashTab[h] != idx) h = (h + 1) % tabSize;
  if(hashTab[h] == -1){
    hashTab[h] = idx;
    return 0;
  } else return 1;
}

static int idxLookup(int idx, int* hashTab, int* idxMap, int tabSize){
  int h = idx % tabSize;
  while(hashTab[h] >= 0 && hashTab[h] != idx) h = (h + 1) % tabSize;
  return idxMap[h];
}

static void idxWrite(int idx, int* hashTab, int* idxMap, int tabSize, int idxLocal){
  int h = idx % tabSize;
  while(hashTab[h] >= 0 && hashTab[h] != idx) h = (h + 1) % tabSize;
  idxMap[h] = idxLocal;
}

static int findroot(Graph* g, int i){
  if (g->ptr[i]<0) return i;
  return g->ptr[i] = findroot(g, g->ptr[i]);
}

static void order_bf_list(Graph* g, int* array, int len){
  int j = len-1;
  for (int i=0; i<len; i++){
   if (!g->parity[findroot(g, array[i])]){
     while(!g->parity[findroot(g, array[j])] && j>i) j--;
     if(j>i){
       int temp = array[i];
       array[i] = array[j];
       array[j] = temp;
     } else break;
   }
  }
}

bool ldpc_decode_cluster(Graph* g, int root){
  int n_row = (- g->ptr[root]) - g->num_qbt[root];
  int n_col = g->num_qbt[root] + 1;

  WMat w = new_wmat(n_row, n_col);
  memset(w.mat, 0, w.num_bytes);
  int* lBf = malloc((- g->ptr[root]) * sizeof(int));
  int tableSize = getNextPrimePower2(2*(- g->ptr[root]));
  int* visited = malloc(sizeof(int)*tableSize);
  int* globalToMat = malloc(sizeof(int)*tableSize);
  for(int i=0; i<tableSize; i++) visited[i] = -1;
  int cntqbtLen = 0;
  int bf_pos = 0;
  int len_lBf = 1;
  lBf[0] = root;
  isVisited(root, visited, tableSize);
  idxWrite(root, visited, globalToMat, tableSize, 0);
  if(g->n_qbt > root){
    g->decode[root] = 0;
    cntqbtLen = 1;
  }
  while(bf_pos < len_lBf){
    int idxLocal = idxLookup(lBf[bf_pos], visited, globalToMat, tableSize);
    uint8_t num_nb_max;
    int* nn;
    int idx_arry;
    if(lBf[bf_pos] >= g->n_qbt){
      if(g->syndrome[lBf[bf_pos] - g->n_qbt]) write_matrix_position_bit(w.mat, idxLocal, n_col - 1, w.num_blocks_row, 1);
      nn = g->nn_syndr;
      num_nb_max = g->num_nb_max_syndr;
      idx_arry = lBf[bf_pos] - g->n_qbt;
    } else {
      g->decode[lBf[bf_pos]] = 0;
      nn = g->nn_qbt;
      num_nb_max = g->num_nb_max_qbt;
      idx_arry = lBf[bf_pos];
    }
    for(uint8_t i=0; i<g->len_nb[lBf[bf_pos]]; i++){
      int nb = nn[idx_arry*num_nb_max + i];
      if(findroot(g, nb) == root){
        int idxLocalNb;
        if(!isVisited(nb, visited, tableSize)){
          if(nb < g->n_qbt) idxLocalNb = cntqbtLen++;
          else idxLocalNb = len_lBf - cntqbtLen;
          idxWrite(nb, visited, globalToMat, tableSize, idxLocalNb);
          lBf[len_lBf++] = nb;
        } else idxLocalNb = idxLookup(nb, visited, globalToMat, tableSize);
        if(lBf[bf_pos] < g->n_qbt) write_matrix_position_bit(w.mat, idxLocalNb, idxLocal, w.num_blocks_row, 1);
      }
    }
    bf_pos++;
  }

  GaussElimin_bit(w.mat, n_row, n_col, n_col);
  bool* decode = malloc(g->num_qbt[root]);
  memset(decode, 0, g->num_qbt[root]);
  bool decodable = true;
  uint64_t num_blocks_row = n_col / (8 * sizeof(dtypeBlk)) + 1;
  for(int r = n_row-1; r>=0; r--){
    if(read_matrix_position_bit(w.mat, r, n_col - 1, num_blocks_row)){
      int c = (r < n_col - 2 ? r : n_col - 2);
      while(!read_matrix_position_bit(w.mat, r, c, num_blocks_row) && c < n_col - 1) c++;
      if(c == n_col - 1) {decodable = false; break;}
      decode[c] = 1;
      for(int r2=r-1; r2>=0; r2--){
        if(read_matrix_position_bit(w.mat, r2, c, num_blocks_row)) flip_matrix_position_bit(w.mat, r2, n_col - 1, num_blocks_row);
      }
    }
  }

  for(int i=0; i<tableSize; i++){
    if(visited[i] >= 0 && g->n_qbt > visited[i] && decode[globalToMat[i]]) g->decode[visited[i]] = 1;
  }

  free_wmat(w);
  free(lBf);
  free(globalToMat);
  free(visited);
  free(decode);
  return decodable;
}

static int merge_root(Graph* g, int r1, int r2){
  if (g->parity[r2]) g->num_invalid -= 1;
  if (g->ptr[r1] > g->ptr[r2]){
    g->num_qbt[r2] += g->num_qbt[r1];
    g->ptr[r2] += g->ptr[r1];
    g->ptr[r1] = r2;
    r1 = r2;
  } else {
    g->num_qbt[r1] += g->num_qbt[r2];
    g->ptr[r1] += g->ptr[r2];
    g->ptr[r2] = r1;
  }
  return r1;
}

static void update_cluster_validity(Graph* g, int root){
  g->parity[root] = !ldpc_decode_cluster(g, root);
  if(!g->parity[root]) g->num_invalid -= 1;
}

struct nodeSk {
  int i_node;
  struct nodeSk* next;
};
typedef struct nodeSk nodeSk;

static void connect_node(nodeSk* node, int i_next){
  nodeSk* next = malloc(sizeof(nodeSk));
  node->next = next;
  next->i_node = i_next;
  next->next = NULL;
}

static void free_nodeSk_list(nodeSk* node){
  while(node->next != NULL){
    nodeSk* h = node;
    node = node->next;
    free(h);
  }
  free(node);
}

/* Algorithm 3: syndrome validation using breadth-first Tanner graph traversal to grow clusters and Gaussian elimination for validating them */
void ldpc_syndrome_validation_and_decode(Graph* g, int num_syndromes){
  memset(g->decode, 0, g->n_qbt * sizeof(bool));
  int nnode = g->n_qbt + g->n_syndr;
  nodeSk** a_skipped = malloc(nnode * sizeof(nodeSk*));
  for(int i=0; i<nnode; i++) a_skipped[i] = NULL;
  nodeSk** a_skipped_last = malloc(nnode * sizeof(nodeSk*));
  int* bf_list = malloc(nnode * sizeof(int));
  int bf_next = 0;
  memset(g->visited, 0, nnode * sizeof(bool));
  for(int i=0; i < g->n_qbt; i++){
    if (g->erasure[i]){
      bf_list[bf_next++] = i;
      g->visited[i] = true;
    }
    g->ptr[i] = -1;
    g->num_qbt[i] = 1;
  }
  int num_erasure = bf_next;
  for(int i=g->n_qbt; i < nnode; i++){
    if (g->syndrome[i - g->n_qbt]){
      bf_list[bf_next++] = i;
      g->visited[i] = true;
    }
    g->ptr[i] = -1;
    g->num_qbt[i] = 0;
  }
  g->num_invalid = num_syndromes;
  int bf_pos = 0;

  while (bf_pos < num_erasure){
    int n = bf_list[bf_pos];
    int r_n = findroot(g, n);
    if (!g->parity[r_n]) g->num_invalid += 1;
    uint8_t num_nb_max;
    int* nn;
    int idx_arry;
    if(n < g->n_qbt){
      num_nb_max = g->num_nb_max_qbt;
      nn = g->nn_qbt;
      idx_arry = n;
    } else {
      num_nb_max = g->num_nb_max_syndr;
      nn = g->nn_syndr;
      idx_arry = n - g->n_qbt;
    }
    for(uint8_t i=0; i<g->len_nb[n]; i++){
      int nb = nn[idx_arry*num_nb_max + i];
      int r_nb = findroot(g, nb);
      if(r_n != r_nb) r_n = merge_root(g, r_n, r_nb);
      if (g->visited[nb] == false) {
        bf_list[bf_next++] = nb;
        g->visited[nb] = true;
      }
    }
    update_cluster_validity(g, r_n);
    bf_pos++;
  }

  order_bf_list(g, bf_list + bf_pos, bf_next - bf_pos);

  while(g->num_invalid > 0){
    int n = bf_list[bf_pos];
    int r_n = findroot(g, n);
    if(g->parity[r_n]){
      for(uint8_t i=0; i<g->len_nb[n]; i++){
        int nb = g->nn_syndr[(n - g->n_qbt)*g->num_nb_max_syndr + i];
        int r_nb = findroot(g, nb);
        if(r_n != r_nb){
          r_n = merge_root(g, r_n, r_nb);
          for(uint8_t j=0; j<g->len_nb[nb]; j++){
            int nb2 = g->nn_qbt[nb*g->num_nb_max_qbt + j];
            int r_nb2 = findroot(g, nb2);
          
            if (a_skipped[r_nb2] != NULL){
              nodeSk* node = a_skipped[r_nb2];
              bf_list[bf_next] = node->i_node;
              bf_next = (bf_next + 1) % nnode;
              while(node->next != NULL){
                node = node->next;
                bf_list[bf_next] = node->i_node;
                bf_next = (bf_next + 1) % nnode;
              }
              free_nodeSk_list(a_skipped[r_nb2]);
              a_skipped[r_nb2] = NULL;
            }

            if(r_n != r_nb2) r_n = merge_root(g, r_n, r_nb2);
            if (g->visited[nb2] == false) {
              bf_list[bf_next] = nb2;
              bf_next = (bf_next + 1) % nnode;
              g->visited[nb2] = true;
            }
          }
        }
      }
      update_cluster_validity(g, r_n);
    } else {
      if (a_skipped[r_n] == NULL){
        a_skipped[r_n] = malloc(sizeof(nodeSk));
        a_skipped[r_n]->i_node = n;
        a_skipped[r_n]->next = NULL;
        a_skipped_last[r_n] = a_skipped[r_n];
      } else {
        connect_node(a_skipped_last[r_n], n);
        a_skipped_last[r_n] = a_skipped_last[r_n]->next;
      }
    }
    bf_pos = (bf_pos + 1) % nnode;
  }

  // // ===== TRACK BULLETPROOF SCAN =====
  // if (g->cluster_sizes != NULL) {
  //   for (int i = 0; i < nnode; i++) {
  //     if (g->ptr[i] < 0) {
  //       bool is_real_cluster = false;
  //       for (int j = 0; j < nnode; j++) {
  //         if (findroot(g, j) == i && g->visited[j]) {
  //           is_real_cluster = true;
  //           break; 
  //         }
  //       }
  //       if (is_real_cluster && g->num_qbt[i] > 0) {
  //         if (g->cluster_count >= g->max_cluster_count) {
  //           g->max_cluster_count *= 2;
  //           g->cluster_sizes = realloc(g->cluster_sizes, g->max_cluster_count * sizeof(int));
  //         }
  //         g->cluster_sizes[g->cluster_count] = g->num_qbt[i];
  //         g->cluster_count++;
  //       }
  //     }
  //   }
  // }

  for(int i=0; i<nnode; i++) if (a_skipped[i] != NULL) free_nodeSk_list(a_skipped[i]);
  free(a_skipped);
  free(a_skipped_last);
  free(bf_list);
}

int check_correction_general(Graph* g){
  memset(g->syndrome, 0, g->n_syndr * sizeof(bool));
  int num_syndromes = 0;
  for(int i=0; i < g->n_qbt; i++){
    if(g->error[i] ^ g->decode[i]){
      for(uint8_t j=0; j<g->len_nb[i]; j++){
        num_syndromes += (g->syndrome[g->nn_qbt[i*g->num_nb_max_qbt+j] - g->n_qbt] ? -1 : +1);
        g->syndrome[g->nn_qbt[i*g->num_nb_max_qbt+j] - g->n_qbt] = (g->syndrome[g->nn_qbt[i*g->num_nb_max_qbt+j] - g->n_qbt] + 1) % 2;
      }
    }
  }
  return num_syndromes;
}

/* given graph and syndrome, compute decoding for general ldpc code */
// void ldpc_collect_graph_and_decode(int n_qbt, int n_syndr, uint8_t num_nb_max_qbt, uint8_t num_nb_max_syndr, int* nn_qbt, int* nn_syndr, uint8_t* len_nb, bool* syndrome, bool* erasure, bool* decode, int* py_cluster_sizes, int* py_cluster_count, int* py_qubit_cluster_map){
//   Graph g;
//   g.n_qbt = n_qbt;
//   g.n_syndr = n_syndr;
//   g.ptr = malloc((n_qbt + n_syndr) * sizeof(int)); 
//   g.num_qbt = malloc((n_qbt + n_syndr) * sizeof(int)); 
//   g.nn_qbt = nn_qbt; 
//   g.nn_syndr = nn_syndr; 
//   g.len_nb = len_nb; 
//   g.num_nb_max_qbt = num_nb_max_qbt; 
//   g.num_nb_max_syndr = num_nb_max_syndr; 
//   g.visited = malloc((n_qbt + n_syndr) * sizeof(bool)); 
//   g.syndrome = syndrome;
//   g.erasure = erasure;
//   g.parity = malloc((n_qbt + n_syndr) * sizeof(bool)); 
//   g.decode = decode; 

//   g.max_cluster_count = 32;
//   g.cluster_sizes = malloc(g.max_cluster_count * sizeof(int));
//   g.cluster_count = 0;

//   memset(g.parity, 0, g.n_qbt * sizeof(bool));
//   memcpy(g.parity + g.n_qbt, g.syndrome, g.n_syndr * sizeof(bool)); 

//   int num_syndrome = 0;
//   for(int i=0; i<g.n_syndr; i++) if(syndrome[i]) num_syndrome++;
  
//   ldpc_syndrome_validation_and_decode(&g, num_syndrome);

//   // Setup local references map tracking structures
//   int nnode = n_qbt + n_syndr;
//   int* root_to_cluster_id = malloc(nnode * sizeof(int));
//   for(int i = 0; i < nnode; i++) root_to_cluster_id[i] = 0;

//   int int_cluster_id = 1; 
  
//   if (g.cluster_sizes != NULL) {
//     for (int i = 0; i < nnode; i++) {
//       if (g.ptr[i] < 0) {
//         bool is_real_cluster = false;
//         for (int j = 0; j < nnode; j++) {
//           if (findroot(&g, j) == i && g.visited[j]) {
//             is_real_cluster = true;
//             break; 
//           }
//         }
//         if (is_real_cluster && g.num_qbt[i] > 0) {
//           if (g.cluster_count >= g.max_cluster_count) {
//             g.max_cluster_count *= 2;
//             g.cluster_sizes = realloc(g.cluster_sizes, g.max_cluster_count * sizeof(int));
//           }
//           g.cluster_sizes[g.cluster_count] = g.num_qbt[i];
          
//           root_to_cluster_id[i] = int_cluster_id;
          
//           g.cluster_count++;
//           int_cluster_id++;
//         }
//       }
//     }
//   }

//   // Populate maps pointers cleanly
//   for (int q = 0; q < n_qbt; q++) {
//     int q_root = findroot(&g, q);
//     py_qubit_cluster_map[q] = root_to_cluster_id[q_root]; 
//   }

//   // Export properties securely back to pre-allocated buffers
//   *py_cluster_count = g.cluster_count;
//   for(int i = 0; i < g.cluster_count; i++) {
//     py_cluster_sizes[i] = g.cluster_sizes[i];
//   }

//   free(root_to_cluster_id);
//   free(g.cluster_sizes);
//   free(g.ptr);
//   free(g.num_qbt);
//   free(g.visited);
//   free(g.parity);
// }

void ldpc_collect_graph_and_decode_batch(int n_qbt, int n_syndr, uint8_t num_nb_max_qbt, uint8_t num_nb_max_syndr, int* nn_qbt, int* nn_syndr, uint8_t* len_nb, bool* syndrome, bool* erasure, bool* decode, int nrep, int* py_cluster_sizes, int* py_cluster_counts, int max_clusters_per_rep, int* py_qubit_cluster_maps){
  Graph g;
  g.n_qbt = n_qbt;
  g.n_syndr = n_syndr;
  int nnode = n_qbt + n_syndr;
  
  // 1. Allocate main structures once
  g.ptr = malloc(nnode * sizeof(int)); 
  g.num_qbt = malloc(nnode * sizeof(int)); 
  g.visited = malloc(nnode * sizeof(bool)); 
  g.parity = malloc(nnode * sizeof(bool)); 
  
  // 2. Allocate the look-up array here, scoped to the function
  int* root_to_cluster_id = malloc(nnode * sizeof(int));

  for(int r=0; r<nrep; r++){
    // Set pointers for current batch
    g.syndrome = syndrome + r*g.n_syndr;
    g.decode = decode + r*g.n_qbt; 
    g.erasure = erasure + r*g.n_qbt;
    g.nn_qbt = nn_qbt; 
    g.nn_syndr = nn_syndr; 
    g.len_nb = len_nb; 
    g.num_nb_max_qbt = num_nb_max_qbt; 
    g.num_nb_max_syndr = num_nb_max_syndr;

    g.max_cluster_count = max_clusters_per_rep > 32 ? max_clusters_per_rep : 32;
    g.cluster_sizes = malloc(g.max_cluster_count * sizeof(int));
    g.cluster_count = 0;

    // 3. SECURE INITIALIZATION: Every shot must clear these
    for(int i = 0; i < nnode; i++) {
        root_to_cluster_id[i] = 0;
        g.visited[i] = false;
        g.ptr[i] = -1;
        g.num_qbt[i] = (i < n_qbt) ? 1 : 0;
    }

    memset(g.parity, 0, g.n_qbt * sizeof(bool));
    memcpy(g.parity + g.n_qbt, g.syndrome, g.n_syndr * sizeof(bool)); 
    
    int num_syndrome = 0;
    for(int i=0; i<g.n_syndr; i++) if(g.syndrome[i]) num_syndrome++;

    if (num_syndrome > 0) {
        ldpc_syndrome_validation_and_decode(&g, num_syndrome);
    }

    // 4. Assign cluster IDs (Strictly scoped to this repetition)
    int int_cluster_id = 1;
    for (int i = 0; i < nnode; i++) {
      if (g.ptr[i] < 0) {
        bool is_real_cluster = false;
        for (int j = 0; j < nnode; j++) {
          if (findroot(&g, j) == i && g.visited[j]) {
            is_real_cluster = true;
            break; 
          }
        }
        if (is_real_cluster && g.num_qbt[i] > 0) {
          if (g.cluster_count < g.max_cluster_count) {
            g.cluster_sizes[g.cluster_count] = g.num_qbt[i];
            g.cluster_count++;
          }
          root_to_cluster_id[i] = int_cluster_id++;
        }
      }
    }

    // 5. Populate maps
    int shot_offset = r * n_qbt;
    for (int q = 0; q < n_qbt; q++) {
        int q_root = findroot(&g, q);
        py_qubit_cluster_maps[shot_offset + q] = root_to_cluster_id[q_root];
    }

    py_cluster_counts[r] = g.cluster_count;
    for(int i = 0; i < g.cluster_count && i < max_clusters_per_rep; i++) {
      py_cluster_sizes[r * max_clusters_per_rep + i] = g.cluster_sizes[i];
    }
    
    free(g.cluster_sizes); // Free per-batch allocation
  }

  // 6. Final cleanup
  free(root_to_cluster_id);
  free(g.ptr);
  free(g.num_qbt);
  free(g.visited);
  free(g.parity);
}

/* given graph and syndrome, compute decoding in batches of nrep repetitions (for general ldpc code) */
void ldpc_collect_graph_and_decode_batch(int n_qbt, int n_syndr, uint8_t num_nb_max_qbt, uint8_t num_nb_max_syndr, int* nn_qbt, int* nn_syndr, uint8_t* len_nb, bool* syndrome, bool* erasure, bool* decode, int nrep, int* py_cluster_sizes, int* py_cluster_counts, int max_clusters_per_rep, int* py_qubit_cluster_maps){
  Graph g;
  g.n_qbt = n_qbt;
  g.n_syndr = n_syndr;
  g.ptr = malloc((n_qbt + n_syndr) * sizeof(int)); 
  g.num_qbt = malloc((n_qbt + n_syndr) * sizeof(int)); 
  g.nn_qbt = nn_qbt; 
  g.nn_syndr = nn_syndr; 
  g.len_nb = len_nb; 
  g.num_nb_max_qbt = num_nb_max_qbt; 
  g.num_nb_max_syndr = num_nb_max_syndr; 
  g.visited = malloc((n_qbt + n_syndr) * sizeof(bool)); 
  g.parity = malloc((n_qbt + n_syndr) * sizeof(bool)); 

  int nnode = n_qbt + n_syndr;
  // Allocate the lookup array securely inside the function frame context
  int* root_to_cluster_id = malloc(nnode * sizeof(int));

  for(int r=0; r<nrep; r++){
    g.syndrome = syndrome + r*g.n_syndr;
    g.decode = decode + r*g.n_qbt; 
    g.erasure = erasure + r*g.n_qbt;

    g.max_cluster_count = max_clusters_per_rep > 32 ? max_clusters_per_rep : 32;
    g.cluster_sizes = malloc(g.max_cluster_count * sizeof(int));
    g.cluster_count = 0;

    // Reset the internal tree structure data layout for this specific repetition run
    for(int i = 0; i < nnode; i++) {
        root_to_cluster_id[i] = 0;
        g.visited[i] = false;
        g.ptr[i] = -1;
        g.num_qbt[i] = (i < n_qbt) ? 1 : 0;
    }

    memset(g.parity, 0, g.n_qbt * sizeof(bool));
    memcpy(g.parity + g.n_qbt, g.syndrome, g.n_syndr * sizeof(bool)); 
    
    int num_syndrome = 0;
    for(int i=0; i<g.n_syndr; i++) if(g.syndrome[i]) num_syndrome++;

    // Only invoke validation if active syndrome flags exist on step zero
    if (num_syndrome > 0) {
        ldpc_syndrome_validation_and_decode(&g, num_syndrome);
    }

    // Assign cluster IDs safely to unique root keys
    int int_cluster_id = 1;
    if (num_syndrome > 0) {
      for (int i = 0; i < nnode; i++) {
        if (g.ptr[i] < 0) {
          bool is_real_cluster = false;
          for (int j = 0; j < nnode; j++) {
            if (findroot(&g, j) == i && g.visited[j]) {
              is_real_cluster = true;
              break; 
            }
          }
          if (is_real_cluster && g.num_qbt[i] > 0) {
            if (g.cluster_count < g.max_cluster_count) {
              g.cluster_sizes[g.cluster_count] = g.num_qbt[i];
              g.cluster_count++;
            }
            root_to_cluster_id[i] = int_cluster_id;
            int_cluster_id++;
          }
        }
      }
    }

    // Export structural membership map matching this shot's offset window
    int shot_offset = r * n_qbt;
    for (int q = 0; q < n_qbt; q++) {
      if (num_syndrome > 0) {
          int q_root = findroot(&g, q);
          py_qubit_cluster_maps[shot_offset + q] = root_to_cluster_id[q_root];
      } else {
          py_qubit_cluster_maps[shot_offset + q] = 0; // Trivial cluster state
      }
    }

    // Package output counts safely
    py_cluster_counts[r] = g.cluster_count;
    for(int i = 0; i < g.cluster_count && i < max_clusters_per_rep; i++) {
      py_cluster_sizes[r * max_clusters_per_rep + i] = g.cluster_sizes[i];
    }
    
    free(g.cluster_sizes);
  }

  free(root_to_cluster_id);
  free(g.ptr);
  free(g.num_qbt);
  free(g.visited);
  free(g.parity);
}