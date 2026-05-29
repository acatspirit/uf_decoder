#ifndef GRAPH_TYPE
#define GRAPH_TYPE

typedef struct {
  int* ptr;
  int* nn_qbt;
  int* nn_syndr;
  uint8_t* len_nb;
  bool* syndrome;
  bool* erasure;
  bool* error;
  bool* parity;
  bool* visited;
  bool* decode;
  int* num_qbt;
  int** logicals;
  int* logical_weight;
  int n_qbt, n_syndr, num_edges, num_invalid, num_logicals;
  uint8_t num_nb_max_qbt, num_nb_max_syndr;

  // gemini suggestions
  int* cluster_sizes;      // Array to store the sizes (n_col) of decoded clusters
  int max_cluster_count;  // Maximum capacity allocated for the cluster_sizes array
  int cluster_count;      // Counter for how many clusters have been recorded
} Graph;

typedef struct {
  int* parent;
  bool* visited;
  bool* leaf;
  int nnode;
} Forest;

Graph new_graph(int n_qbt, uint8_t num_nb_max_qbt, int n_syndr, uint8_t num_nb_max_syndr);
void free_graph(Graph* g);
Forest new_forest(int nnode);
void free_forest(Forest* f);

#endif
