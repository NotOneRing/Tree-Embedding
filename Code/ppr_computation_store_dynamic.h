extern "C"
{
#include "matrix_vector_functions_intel_mkl.h"
#include "matrix_vector_functions_intel_mkl_ext.h"
#include "string.h"
}

#undef max
#undef min

#include <algorithm>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <deque>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <cstring>
#include <thread>
#include <mutex>
#include "Eigen/Sparse"
#include "Eigen/Dense"
#include <chrono>
#include <climits>

#include<queue>
#include "my_queue.h"

#include<assert.h>
#include <unordered_map>
#include<cmath>

#include<list>

#include<memory.h>

#include <atomic>


#include"Graph_dynamic.h"


using namespace Eigen;

using namespace std;


class column_tuple{
  public:
    int row;
    float pi;
    column_tuple(int row, float pi){
      this->row = row;
      this->pi = pi;
    }
};




class sparse_d_row_tree_mkl{
  public:
  int nParts;

  int level_p;
  int total_nodes;
  vector<mat*> matrix_vec;
  int largest_level_start_index;
  int largest_level_end_index;
  vector<mat*> hierarchy_matrix_vec;

  Eigen::VectorXd vS_cur_iter;

  Eigen::MatrixXd U_cur_iter;

  vector<mat*> near_n_matrix_vec;
  vector<mat*> less_near_n;

  unordered_map<int, SparseMatrix<double, 0, int64_t>> mat_mapping;

  // unordered_map<int, MatrixXd> dense_mat_mapping;

  int start_row;
  int end_row;
  int hierarchy_n;
  int row_dim;

  vector<double> norm_B_Bid_difference_vec;


  sparse_d_row_tree_mkl(int row_dim, int d, int nParts, int hierarchy_n, unordered_map<int, vector<int>> &vec_mapping,
  int start_row, int end_row){

    this->nParts = nParts;

    this->row_dim = row_dim;

    this->start_row = start_row;
    this->end_row = end_row;
    this->hierarchy_n = hierarchy_n;


    norm_B_Bid_difference_vec.resize(vec_mapping.size());

    for(int i = 0; i < vec_mapping.size(); i++){
      vector<int> &v = vec_mapping[i];
      int current_group_size = vec_mapping[i].size();
      mat_mapping[i].resize(row_dim, current_group_size);

      norm_B_Bid_difference_vec[i] = 0;
    }


    level_p = log(nParts) / log(hierarchy_n) + 1;

    assert(pow(hierarchy_n, level_p - 1) == nParts);

    total_nodes = (pow(hierarchy_n, level_p) - 1) / (hierarchy_n - 1);

    largest_level_start_index = total_nodes - nParts;

    largest_level_end_index = total_nodes - 1;

    hierarchy_matrix_vec.resize(total_nodes);

    for(int i = 0; i < total_nodes; i++){
      hierarchy_matrix_vec[i] = matrix_new(row_dim, d);
    }

    near_n_matrix_vec.resize(total_nodes - nParts);

    less_near_n.resize(total_nodes - nParts);
  }
  
};













class d_row_tree_mkl{
  public:
  int nParts;

  int level_p;
  int total_nodes;
  vector<mat*> matrix_vec;
  int largest_level_start_index;
  int largest_level_end_index;
  vector<mat*> hierarchy_matrix_vec;

  Eigen::VectorXd vS_cur_iter;

  Eigen::MatrixXd U_cur_iter;

  vector<mat*> near_n_matrix_vec;
  vector<mat*> less_near_n;

  unordered_map<int, SparseMatrix<double, 0, int>> mat_mapping;

  unordered_map<int, MatrixXd> dense_mat_mapping;

  int start_row;
  int end_row;
  int hierarchy_n;
  int row_dim;

  vector<double> norm_B_Bid_difference_vec;


  d_row_tree_mkl(int row_dim, int d, int nParts, int hierarchy_n, unordered_map<int, vector<int>> &vec_mapping,
  int start_row, int end_row){

    this->nParts = nParts;

    this->row_dim = row_dim;

    this->start_row = start_row;
    this->end_row = end_row;
    this->hierarchy_n = hierarchy_n;


    norm_B_Bid_difference_vec.resize(vec_mapping.size());

    for(int i = 0; i < vec_mapping.size(); i++){
      vector<int> &v = vec_mapping[i];
      int current_group_size = vec_mapping[i].size();
      mat_mapping[i].resize(row_dim, current_group_size);
      dense_mat_mapping[i].resize(row_dim, current_group_size);
      norm_B_Bid_difference_vec[i] = 0;
    }


    level_p = log(nParts) / log(hierarchy_n) + 1;

    assert(pow(hierarchy_n, level_p - 1) == nParts);

    total_nodes = (pow(hierarchy_n, level_p) - 1) / (hierarchy_n - 1);

    cout<<"total_nodes = "<<total_nodes<<endl;

    largest_level_start_index = total_nodes - nParts;

    cout<<"largest_level_start_index = "<<largest_level_start_index<<endl;
    
    largest_level_end_index = total_nodes - 1;
    
    cout<<"largest_level_end_index = "<<largest_level_end_index<<endl;
    
    hierarchy_matrix_vec.resize(total_nodes);

    for(int i = 0; i < total_nodes; i++){
      hierarchy_matrix_vec[i] = matrix_new(row_dim, d);
    }

    near_n_matrix_vec.resize(total_nodes - nParts);

    less_near_n.resize(total_nodes - nParts);
  }
  
};



class d_row_tree{
  public:

  int level_p;
  int total_nodes;
  vector<MatrixXd> matrix_vec;
  int largest_level_start_index;
  int largest_level_end_index;
  vector<MatrixXd> hierarchy_matrix_vec;

  Eigen::VectorXd vS_cur_iter;

  Eigen::MatrixXd U_cur_iter;

  vector<Eigen::MatrixXd> near_n_matrix_vec;
  vector<MatrixXd> less_near_n;

  unordered_map<int, SparseMatrix<double, 0, int>> mat_mapping;
  unordered_map<int, MatrixXd> dense_mat_mapping;

  int start_row;
  int end_row;
  int hierarchy_n;
  int row_dim;

  d_row_tree(int row_dim, int d, int nParts, int hierarchy_n, unordered_map<int, vector<int>> &vec_mapping,
  int start_row, int end_row){

    this->row_dim = row_dim;

    this->start_row = start_row;
    this->end_row = end_row;
    this->hierarchy_n = hierarchy_n;
    
    for(int i = 0; i < vec_mapping.size(); i++){
      vector<int> &v = vec_mapping[i];
      int current_group_size = vec_mapping[i].size();
      mat_mapping[i].resize(row_dim, current_group_size);
      dense_mat_mapping[i].resize(row_dim, current_group_size);
    }


    level_p = log(nParts) / log(hierarchy_n) + 1;

    assert(pow(hierarchy_n, level_p - 1) == nParts);

    total_nodes = (pow(hierarchy_n, level_p) - 1) / (hierarchy_n - 1);

    largest_level_start_index = total_nodes - nParts;

    largest_level_end_index = total_nodes - 1;

    hierarchy_matrix_vec.resize(total_nodes);

    for(int i = 0; i < total_nodes; i++){
      hierarchy_matrix_vec[i].resize(row_dim, d);
    }

    near_n_matrix_vec.resize(total_nodes - nParts);

    less_near_n.resize(total_nodes - nParts);
  }
  
};



void DynamicForwardPush(int start, int end, UGraph* g, double residuemax, double reservemin, vector<Triplet<double>>* answer, 
long long int& all_count, double alpha, vector<int>& labeled_node_vec, 
double** residue, 
double** pi,
bool** flags, 
Queue* queue_list,
vector<vector<int>> &update_mat_record,
int row_dim,
int col_dim,
int number_of_d_row_tree,
int nParts
)
{

  int vertices_number = g->n;  

  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);

      if(g->degree[v] == 0){
        flags[it][v] = false;
        queue_list[it].front++;
        continue;
      }

      if(residue[it][v] / g->degree[v] > residuemax){
        for(int j = 0; j < g->degree[v]; j++){
          int u = g->AdjList[v][j];
          residue[it][u] += (1-alpha) * residue[it][v] / g->degree[v];

          if(g->degree[u] == 0){
            continue;
          }
          
          if(residue[it][u] / g->degree[u] > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }
        }
        pi[it][v] += alpha * residue[it][v];

        double changed_value = alpha * residue[it][v];
        
        answer->push_back(Triplet<double>(v, src, changed_value));
        answer->push_back(Triplet<double>(src, v, changed_value));


        int row_v = it / row_dim;
        int col_v = v / col_dim;
        if(row_v == number_of_d_row_tree){
          row_v--;
        }
        if(col_v == nParts){
          col_v--;
        }

        if(update_mat_record[row_v][col_v] != 0){
          update_mat_record[row_v][col_v] = 0;
        }

        residue[it][v] = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }


    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }
  return;

}



















































void ForwardPushSymmetric(int start, int end, UGraph* g, double residuemax, double reservemin, vector<Triplet<double>>* answer, 
long long int& all_count, double alpha, vector<int>& labeled_node_vec){

  cout<<"Start ForwardPush!"<<endl;
  cout<<"start = "<<start<<endl;
  cout<<"end = "<<end<<endl;

  int vertices_number = g->n;

  double *residue = new double[vertices_number];

  double *pi = new double[vertices_number];

  bool *flags = new bool[vertices_number];

  int upper_nnz = ceil(1 / residuemax / alpha);



  //Create Queue
	Queue Q =
	{
		//.arr = 
		(int*)malloc( sizeof(int) * (upper_nnz + 2) * 2 ),
		//.capacity = 
		(upper_nnz + 2) * 2,
		//.front =
		0,
		//.rear = 
		0
	};

  //The queue to record used elements in this node
	Queue record_Q =
	{
		//.arr = 
		(int*)malloc(sizeof(int) * (upper_nnz + 2) * 2 ),
		//.capacity = 
		(upper_nnz + 2) * 2,
		//.front =
		0,
		//.rear = 
		0
	};




  memset(residue, 0, sizeof(double) * vertices_number);

  memset(pi, 0, sizeof(double) * vertices_number);

  memset(flags, false, sizeof(bool) * vertices_number);

  int record_max;
  
  for(int it = start; it < end; it++){


    Q.front = 0;
    Q.rear = 0;

    int src = labeled_node_vec[it];


    while(!isEmpty(&record_Q)){
      int current = get_front(&record_Q);
      residue[current] = 0;
      flags[current] = false;
      pi[current] = 0;
      dequeue(&record_Q);
    }

    record_Q.front = 0;
    record_Q.rear = 0;


    residue[src] = 1;
    flags[src] = true;
    
    enqueue(&Q, src);



    while(!isEmpty(&Q)){

      int v = get_front(&Q);
      

      if(g->degree[v] == 0){
        flags[v] = false;
        Q.front++;

        continue;
      }

      if(residue[v] / g->degree[v] > residuemax){
        for(int j = 0; j < g->degree[v]; j++){
          int u = g->AdjList[v][j];
          residue[u] += (1-alpha) * residue[v] / g->degree[v];
          
          enqueue(&record_Q, u);

          if(g->degree[u] == 0){
            continue;
          }
          
          if(residue[u] / g->degree[u] > residuemax && !flags[u]){
            
            enqueue(&Q, u);
            flags[u] = true;
          }
        }
        pi[v] += alpha * residue[v];
        residue[v] = 0;
      }
      
      flags[v] = false;

      Q.front++;
    }




    for(int i = 0; i < Q.rear; i++){
      int index = Q.arr[i];
      if(pi[index] != 0){ 
        answer->push_back(Triplet<double>(index, src, pi[index]));
        answer->push_back(Triplet<double>(src, index, pi[index]));
        all_count += 1;
      }
    }


  }



  delete[] residue;
  delete[] pi;
  delete[] flags;

  residue = NULL;
  pi = NULL;
  flags = NULL;

  if (NULL != Q.arr)
    {
        free(Q.arr);
        Q.arr = NULL;
    }

  if (NULL != record_Q.arr)
    {
        free(record_Q.arr);
        record_Q.arr = NULL;
    }

  cout<<"End ForwardPush!"<<endl;
  return;

}


























void DirectedDynamicForwardPush(int start, int end, Graph* g, double residuemax, double reservemin, vector<Triplet<double>>* answer, 
long long int& all_count, double alpha, vector<int>& labeled_node_vec, 
double** residue, 
double** pi,
bool** flags, 
Queue* queue_list,
vector<vector<int>> &update_mat_record,
int row_dim,
int col_dim,
int number_of_d_row_tree,
int nParts
)
{
  int vertices_number = g->n;  

  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);

      if(g->outdegree[v] == 0){
        flags[it][v] = false;
        queue_list[it].front++;

        continue;
      }

      if(residue[it][v] / g->outdegree[v] > residuemax){
        for(int j = 0; j < g->outdegree[v]; j++){
          int u = g->outAdjList[v][j];
          residue[it][u] += (1-alpha) * residue[it][v] / g->outdegree[v];

          if(g->outdegree[u] == 0){
            continue;
          }
          
          if(residue[it][u] / g->outdegree[u] > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }
        }
        pi[it][v] += alpha * residue[it][v];

        double changed_value = alpha * residue[it][v];
        
        answer->push_back(Triplet<double>(src, v, changed_value));


        int row_v = it / row_dim;
        int col_v = v / col_dim;
        if(row_v == number_of_d_row_tree){
          row_v--;
        }
        if(col_v == nParts){
          col_v--;
        }

        if(update_mat_record[row_v][col_v] != 0){
          update_mat_record[row_v][col_v] = 0;
        }

        residue[it][v] = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }



    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }


  return;

}























void DirectedDynamicForwardPushTranspose(int start, int end, Graph* g, double residuemax, double reservemin, 
vector<Triplet<double>>* answer, 
long long int& all_count, double alpha, vector<int>& labeled_node_vec, 
double** residue, 
double** pi,
bool** flags, 
Queue* queue_list,
vector<vector<int>> &update_mat_record,
int row_dim,
int col_dim,
int number_of_d_row_tree,
int nParts
)
{


  int vertices_number = g->n;  

  
  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);

      if(g->indegree[v] == 0){
        flags[it][v] = false;
        queue_list[it].front++;

        continue;
      }

      if(residue[it][v] / g->indegree[v] > residuemax){
        for(int j = 0; j < g->indegree[v]; j++){
          int u = g->inAdjList[v][j];
          residue[it][u] += (1-alpha) * residue[it][v] / g->indegree[v];

          if(g->indegree[u] == 0){
            continue;
          }
          
          if(residue[it][u] / g->indegree[u] > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }
        }
        pi[it][v] += alpha * residue[it][v];
        double changed_value = alpha * residue[it][v];

        answer->push_back(Triplet<double>(src, v, changed_value));


        int row_v = it / row_dim;
        int col_v = v / col_dim;
        if(row_v == number_of_d_row_tree){
          row_v--;
        }
        if(col_v == nParts){
          col_v--;
        }

        if(update_mat_record[row_v][col_v] != 0){
          update_mat_record[row_v][col_v] = 0;

        }

        residue[it][v] = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }

    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }

  return;

}



















void Directed_Refresh_PPR_initialization(int start, int end, Graph* g, double residuemax, double reservemin,
vector<Triplet<double>>* answer, 
double alpha, vector<int>& labeled_node_vec, 

double** residue, 
double** pi,
bool** flags, 
Queue* queue_list,
double** residue_transpose, 
double** pi_transpose,
bool** flags_transpose, 
Queue * queue_list_transpose,

int iter,
int vertex_number){



  if(iter == 1){
    for(int i = start; i < end; i++){
      queue_list[i].front = 0;
      queue_list[i].rear = 0;

      int src = labeled_node_vec[i];
      residue[i][src] = 1;
      flags[i][src] = true;
      enqueue(&queue_list[i], src);
    }
  }
  else{

    for(int i = 0; i < vertex_number; i++){
      int from_node = i;
      for(int j = g->former_outdegree[from_node]; j < g->outdegree[from_node]; j++){
        int to_node = g->outAdjList[from_node][j];
        
        for(int k = start; k < end; k++){
          if(j == 0){
            continue;
            if(residue[k][from_node] / j > residuemax && !flags[k][from_node]){
              enqueue(&queue_list[k], from_node);
              flags[k][from_node] = true;
            }
          }

          double changed_value = pi[k][from_node] * 1 / j;
          pi[k][from_node] *= (j + 1) / j;
                
          answer->push_back(Triplet<double>(labeled_node_vec[k], from_node, changed_value));

          residue[k][from_node] -= pi[k][from_node] / (j+1) / alpha;
          residue[k][to_node] += (1 - alpha) * pi[k][from_node] / (j+1) / alpha;
          if(residue[k][from_node] / j > residuemax && !flags[k][from_node]){
            enqueue(&queue_list[k], from_node);
            flags[k][from_node] = true;
          }
          if(residue[k][to_node] / j > residuemax && !flags[k][to_node]){
            enqueue(&queue_list[k], to_node);
            flags[k][to_node] = true;
          }
        }

      }
    }
  }



  if(iter == 1){
    
    for(int i = start; i < end; i++){
      queue_list_transpose[i].front = 0;
      queue_list_transpose[i].rear = 0;

      int src = labeled_node_vec[i];
      residue_transpose[i][src] = 1;
      flags_transpose[i][src] = true;
      enqueue(&queue_list_transpose[i], src);
    }

  }
  else{

    for(int i = 0; i < vertex_number; i++){
      int from_node = i;
      for(int j = g->former_indegree[from_node]; j < g->indegree[from_node]; j++){
        int to_node = g->inAdjList[from_node][j];

        for(int k = start; k < end; k++){

          if(j == 0){
            continue;
            if(residue_transpose[k][from_node] / j > residuemax && !flags_transpose[k][from_node]){
              enqueue(&queue_list_transpose[k], from_node);
              flags_transpose[k][from_node] = true;
            }

          }

          double changed_value = pi_transpose[k][from_node] * 1 / j;

          answer->push_back(Triplet<double>(labeled_node_vec[k], from_node, changed_value));

          pi_transpose[k][from_node] *= (j + 1) / j;
          residue_transpose[k][from_node] -= pi_transpose[k][from_node] / (j+1) / alpha;
          residue_transpose[k][to_node] += (1 - alpha) * pi_transpose[k][from_node] / (j+1) / alpha;
          if(residue_transpose[k][from_node] / j > residuemax && !flags_transpose[k][from_node]){
            enqueue(&queue_list_transpose[k], from_node);
            flags_transpose[k][from_node] = true;
          }
          if(residue_transpose[k][to_node] / j > residuemax && !flags_transpose[k][to_node]){
            enqueue(&queue_list_transpose[k], to_node);
            flags_transpose[k][to_node] = true;
          }
        }
        
      }
    }
  }


}














void Undirected_Refresh_PPR_initialization(int start, int end, UGraph* g, double residuemax, double reservemin,
vector<Triplet<double>>* answer, 
double alpha, vector<int>& labeled_node_vec, 

double** residue, 
double** pi,
bool** flags, 
Queue* queue_list,
int iter,
int vertex_number

){

  if(iter == 1){
    for(int i = start; i < end; i++){
      queue_list[i].front = 0;
      queue_list[i].rear = 0;

      int src = labeled_node_vec[i];
      residue[i][src] = 1;
      flags[i][src] = true;
      enqueue(&queue_list[i], src);
    }
  }
  else{

    for(int i = 0; i < vertex_number; i++){
      int from_node = i;
      for(int j = g->former_degree[from_node]; j < g->degree[from_node]; j++){
        int to_node = g->AdjList[from_node][j];
        for(int k = start; k < end; k++){

          if(j == 0){
            continue;
            if(residue[k][from_node] / j > residuemax && !flags[k][from_node]){
              enqueue(&queue_list[k], from_node);
              flags[k][from_node] = true;
            }
          }
          
          double changed_value = pi[k][from_node] * 1 / j;
          answer->push_back(Triplet<double>(labeled_node_vec[k], from_node, changed_value));
          answer->push_back(Triplet<double>(from_node, labeled_node_vec[k], changed_value));

          pi[k][from_node] *= (j + 1) / j;

          
          residue[k][from_node] -= pi[k][from_node] / (j+1) / alpha;
          residue[k][to_node] += (1 - alpha) * pi[k][from_node] / (j+1) / alpha;
          if(residue[k][from_node] / j > residuemax && !flags[k][from_node]){
            enqueue(&queue_list[k], from_node);
            flags[k][from_node] = true;
          }
          if(residue[k][to_node] / j > residuemax && !flags[k][to_node]){
            enqueue(&queue_list[k], to_node);
            flags[k][to_node] = true;
          }
        }
      }
    }
  }

}
































































































































































































































void DenseDynamicForwardPush(int start, int end, UGraph* g, double residuemax, 
double alpha, 
vector<int>& labeled_node_vec, 
float** residue, 
float** pi,

bool** flags, 
Queue* queue_list,

unordered_map<int, int> &row_index_mapping,

int col_dim,

int nParts,
d_row_tree_mkl* subset_tree,
vector<int> &inner_group_mapping,
vector<int> &indicator

)
{

  int vertices_number = g->n;  

  
  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);


      if(g->degree[v] == 0){
        flags[it][v] = false;


        pi[it][v] = alpha * residue[it][v];

        queue_list[it].front++;


        continue;
      }


      if(residue[it][v] / g->degree[v] > residuemax){
        for(int j = 0; j < g->degree[v]; j++){
          int u = g->AdjList[v][j];
          residue[it][u] += (1-alpha) * residue[it][v] / g->degree[v];


          if(g->degree[u] == 0){

            pi[it][u] += alpha * residue[it][v];
            residue[it][src] += (1-alpha) * residue[it][v];
            continue;
          }
          
          if(residue[it][u] / g->degree[u] > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }

        }
        pi[it][v] += alpha * residue[it][v];


        int col_v = v / col_dim;
        
        if(col_v == nParts){
          col_v--;
        }

        int inner_col_index = inner_group_mapping[v];


        subset_tree->dense_mat_mapping[col_v](
          it, inner_col_index) += alpha * residue[it][v];

        residue[it][v] = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }


    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }

  return;

}

















void nodegree_DenseDynamicForwardPush(int start, int end, UGraph* g, double residuemax, 
double alpha, 
vector<int>& labeled_node_vec, 
float** residue, 
float** pi,


bool** flags, 
Queue* queue_list,

unordered_map<int, int> &row_index_mapping,


int col_dim,

int nParts,


d_row_tree_mkl* subset_tree,
vector<int> &inner_group_mapping,
vector<int> &indicator

)
{

  int vertices_number = g->n;  

  
  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);


      if(g->degree[v] == 0){
        flags[it][v] = false;


        pi[it][v] = alpha * residue[it][v];

        queue_list[it].front++;


        continue;
      }



      if(residue[it][v] > residuemax){
        for(int j = 0; j < g->degree[v]; j++){
          int u = g->AdjList[v][j];
          residue[it][u] += (1-alpha) * residue[it][v] / g->degree[v];


          if(g->degree[u] == 0){

            pi[it][u] += alpha * residue[it][v];
            residue[it][src] += (1-alpha) * residue[it][v];
            continue;
          }
          

          if(residue[it][u] > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }

        }
        pi[it][v] += alpha * residue[it][v];




        int col_v = v / col_dim;
        
        if(col_v == nParts){
          col_v--;
        }

        int inner_col_index = inner_group_mapping[v];


        subset_tree->dense_mat_mapping[col_v](
          it, inner_col_index) += alpha * residue[it][v];

        residue[it][v] = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }


    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }

  return;

}



























void nodegree_DenseDynamicForwardPush_LP(int start, int end, UGraph* g, double residuemax, 
double alpha, 
vector<int>& labeled_node_vec, 
double** residue, 
double** pi,

bool** flags, 
Queue* queue_list,

unordered_map<int, int> &row_index_mapping,

int col_dim,
int nParts,

d_row_tree_mkl* subset_tree,
vector<int> &inner_group_mapping,
vector<int> &indicator

)
{

  int vertices_number = g->n;  

  
  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);


      if(g->degree[v] == 0){
        flags[it][v] = false;


        pi[it][v] = alpha * residue[it][v];

        queue_list[it].front++;

        continue;
      }


      if(residue[it][v] > residuemax){
        for(int j = 0; j < g->degree[v]; j++){
          int u = g->AdjList[v][j];
          residue[it][u] += (1-alpha) * residue[it][v] / g->degree[v];


          if(g->degree[u] == 0){

            pi[it][u] += alpha * residue[it][v];
            residue[it][src] += (1-alpha) * residue[it][v];
            continue;
          }
          

          if(residue[it][u] > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }

        }
        pi[it][v] += alpha * residue[it][v];


        int col_v = v / col_dim;
        
        if(col_v == nParts){
          col_v--;
        }

        int inner_col_index = inner_group_mapping[v];


        subset_tree->dense_mat_mapping[col_v](
          it, inner_col_index) += alpha * residue[it][v];

        residue[it][v] = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }


    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }

  return;

}






































void DenseDynamicForwardPush_LP(int start, int end, UGraph* g, double residuemax, 
double alpha, 
vector<int>& labeled_node_vec, 
double** residue, 
double** pi,

bool** flags, 
Queue* queue_list,

unordered_map<int, int> &row_index_mapping,

int col_dim,
int nParts,

d_row_tree_mkl* subset_tree,
vector<int> &inner_group_mapping,
vector<int> &indicator

)
{

  int vertices_number = g->n;  

  
  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);


      if(g->degree[v] == 0){
        flags[it][v] = false;

        pi[it][v] = alpha * residue[it][v];

        queue_list[it].front++;


        continue;
      }


      if(residue[it][v] / g->degree[v] > residuemax){
        for(int j = 0; j < g->degree[v]; j++){
          int u = g->AdjList[v][j];
          residue[it][u] += (1-alpha) * residue[it][v] / g->degree[v];


          if(g->degree[u] == 0){

            pi[it][u] += alpha * residue[it][v];
            residue[it][src] += (1-alpha) * residue[it][v];
            continue;
          }
          
          if(residue[it][u] / g->degree[u] > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }

        }
        pi[it][v] += alpha * residue[it][v];


        int col_v = v / col_dim;
        
        if(col_v == nParts){
          col_v--;
        }

        int inner_col_index = inner_group_mapping[v];


        subset_tree->dense_mat_mapping[col_v](
          it, inner_col_index) += alpha * residue[it][v];

        residue[it][v] = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }


    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }

  return;

}
















//undirected version
void DenseDynamicForwardPush_TransposeRefresh(int start, int end, 
vector<vector<column_tuple*>>& pi_transpose_storepush,
vector<vector<int>> &update_mat_record,
int row_dim,
int col_dim,
int number_of_d_row_tree,
int nParts,
vector<d_row_tree_mkl*> &d_row_tree_vec,
vector<int> &inner_group_mapping,
vector<int>& labeled_node_vec
)
{
    for(int i = start; i < end; i++){
        for(int j = 0; j < pi_transpose_storepush[i].size(); j++){
            int row = pi_transpose_storepush[i][j]->row;
            int col = labeled_node_vec[i];
            float pi = pi_transpose_storepush[i][j]->pi;

            int row_v = row / row_dim;
            int col_v = col / col_dim;

            if(row_v == number_of_d_row_tree){
                row_v--;
            }
            if(col_v == nParts){
                col_v--;
            }

            int inner_col_index = inner_group_mapping[col];

            (d_row_tree_vec[row_v])->dense_mat_mapping[col_v](
                row - row_v * row_dim, inner_col_index) += pi;

        }
    }

}
















//undirected version
void DenseDynamicForwardPush_TransposeRefresh_LP(int start, int end, 
vector<vector<column_tuple*>>& pi_transpose_storepush,
vector<vector<int>> &update_mat_record,
int row_dim,
int col_dim,
int number_of_d_row_tree,
int nParts,
vector<d_row_tree_mkl*> &d_row_tree_vec,
vector<int> &inner_group_mapping,
vector<int>& labeled_node_vec,
vector<MatrixXd> & transpose_trMat_vec
)
{
    for(int i = start; i < end; i++){
        for(int j = 0; j < pi_transpose_storepush[i].size(); j++){
            int row = pi_transpose_storepush[i][j]->row;

            int col = labeled_node_vec[i];

            float pi = pi_transpose_storepush[i][j]->pi;

            int row_v = row / row_dim;
            int col_v = col / col_dim;
            
            
            if(row_v == number_of_d_row_tree){
                row_v--;
            }
            if(col_v == nParts){
                col_v--;
            }

            int inner_col_index = inner_group_mapping[col];

            (d_row_tree_vec[row_v])->dense_mat_mapping[col_v](
                row - row_v * row_dim, inner_col_index) += pi;
            
            transpose_trMat_vec[row_v](col, row) += pi;

        }
    }

}
































































void DenseUndirected_Refresh_PPR_initialization(int start, int end, UGraph* g, double residuemax, 
double alpha, vector<int>& labeled_node_vec, 
float** residue, 
float** pi,
bool** flags, 
Queue* queue_list,
int iter,
int vertex_number,
int col_dim,
int nParts,
vector<int> &inner_group_mapping,
d_row_tree_mkl* subset_tree,
unordered_map<int, int> &row_index_mapping,
vector<int> &indicator,
int dynamic_ppr_start_iter
){

    if(iter < dynamic_ppr_start_iter){
      for(int i = start; i < end; i++){
        memset(residue[i], 0, sizeof(float) * vertex_number);
        memset(pi[i], 0, sizeof(float) * vertex_number);
        memset(flags[i], false, sizeof(bool) * vertex_number);
        queue_list[i].front = 0;
        queue_list[i].rear = 0;

        int src = labeled_node_vec[i];
        residue[i][src] = 1;
        flags[i][src] = true;
        enqueue(&queue_list[i], src);
      }
    }
    else{

    for(int i = 0; i < vertex_number; i++){
        int from_node = i;
        for(int j = g->former_degree[from_node]; j < g->degree[from_node]; j++){
        int to_node = g->AdjList[from_node][j];
        for(int k = start; k < end; k++){

            if(j == 0){
              continue;
            if(residue[k][from_node] / j > residuemax && !flags[k][from_node]){
                enqueue(&queue_list[k], from_node);
                flags[k][from_node] = true;
            }

            }

            int col_v = from_node / col_dim;

            if(col_v == nParts){
            col_v--;
            }


            int inner_col_index = inner_group_mapping[from_node];

            subset_tree->dense_mat_mapping[col_v](
            k, inner_col_index) += pi[k][from_node] * 1 / j;


            pi[k][from_node] *= (j + 1) / j;

            residue[k][from_node] -= pi[k][from_node] / (j+1) / alpha;
            residue[k][to_node] += (1 - alpha) * pi[k][from_node] / (j+1) / alpha;
            if(residue[k][from_node] / j > residuemax && !flags[k][from_node]){
            enqueue(&queue_list[k], from_node);
            flags[k][from_node] = true;
            }
            if(residue[k][to_node] / j > residuemax && !flags[k][to_node]){
            enqueue(&queue_list[k], to_node);
            flags[k][to_node] = true;
            }
        }
        }
    }
    }

}



















void nodegree_DenseUndirected_Refresh_PPR_initialization(int start, int end, UGraph* g, double residuemax, 
double alpha, vector<int>& labeled_node_vec, 
float** residue, 
float** pi,
bool** flags, 
Queue* queue_list,
int iter,
int vertex_number,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,

d_row_tree_mkl* subset_tree,

unordered_map<int, int> &row_index_mapping,
vector<int> &indicator,
int dynamic_ppr_start_iter
){

    if(iter < dynamic_ppr_start_iter){
      for(int i = start; i < end; i++){
        memset(residue[i], 0, sizeof(float) * vertex_number);
        memset(pi[i], 0, sizeof(float) * vertex_number);
        memset(flags[i], false, sizeof(bool) * vertex_number);
        queue_list[i].front = 0;
        queue_list[i].rear = 0;

        int src = labeled_node_vec[i];
        residue[i][src] = 1;
        flags[i][src] = true;
        enqueue(&queue_list[i], src);
      }
    }
    else{

    for(int i = 0; i < vertex_number; i++){
        int from_node = i;
        for(int j = g->former_degree[from_node]; j < g->degree[from_node]; j++){
          int to_node = g->AdjList[from_node][j];
          for(int k = start; k < end; k++){

              if(j == 0){
                continue;
                
              if(residue[k][from_node] > residuemax && !flags[k][from_node]){
                  enqueue(&queue_list[k], from_node);
                  flags[k][from_node] = true;
              }

              }

              int col_v = from_node / col_dim;

              if(col_v == nParts){
              col_v--;
              }


              int inner_col_index = inner_group_mapping[from_node];

              subset_tree->dense_mat_mapping[col_v](
              k, inner_col_index) += pi[k][from_node] * 1 / j;



              pi[k][from_node] *= (j + 1) / j;

              residue[k][from_node] -= pi[k][from_node] / (j+1) / alpha;
              residue[k][to_node] += (1 - alpha) * pi[k][from_node] / (j+1) / alpha;

              if(residue[k][from_node] > residuemax && !flags[k][from_node]){
              enqueue(&queue_list[k], from_node);
              flags[k][from_node] = true;
              }

              if(residue[k][to_node] > residuemax && !flags[k][to_node]){
              enqueue(&queue_list[k], to_node);
              flags[k][to_node] = true;
              }
          }
        }
    }
    }

}





















void nodegree_DenseUndirected_Refresh_PPR_initialization_LP(int start, int end, UGraph* g, double residuemax, 
double alpha, vector<int>& labeled_node_vec, 
double** residue, 
double** pi,
bool** flags, 
Queue* queue_list,
int iter,
int vertex_number,
int col_dim,

int nParts,
vector<int> &inner_group_mapping,


d_row_tree_mkl* subset_tree,

unordered_map<int, int> &row_index_mapping,
vector<int> &indicator,
int dynamic_ppr_start_iter
){

    if(iter < dynamic_ppr_start_iter){
      for(int i = start; i < end; i++){
        memset(residue[i], 0, sizeof(float) * vertex_number);
        memset(pi[i], 0, sizeof(float) * vertex_number);
        memset(flags[i], false, sizeof(bool) * vertex_number);
        queue_list[i].front = 0;
        queue_list[i].rear = 0;

        int src = labeled_node_vec[i];
        residue[i][src] = 1;
        flags[i][src] = true;
        enqueue(&queue_list[i], src);
      }
    }
    else{

    for(int i = 0; i < vertex_number; i++){
        int from_node = i;
        for(int j = g->former_degree[from_node]; j < g->degree[from_node]; j++){
        int to_node = g->AdjList[from_node][j];
        for(int k = start; k < end; k++){

            if(j == 0){
              continue;

            if(residue[k][from_node] > residuemax && !flags[k][from_node]){
                enqueue(&queue_list[k], from_node);
                flags[k][from_node] = true;
            }

            }

            

            int col_v = from_node / col_dim;
            
            if(col_v == nParts){
            col_v--;
            }


            int inner_col_index = inner_group_mapping[from_node];

            subset_tree->dense_mat_mapping[col_v](
            k, inner_col_index) += pi[k][from_node] * 1 / j;



            pi[k][from_node] *= (j + 1) / j;

            residue[k][from_node] -= pi[k][from_node] / (j+1) / alpha;
            residue[k][to_node] += (1 - alpha) * pi[k][from_node] / (j+1) / alpha;

            if(residue[k][from_node] > residuemax && !flags[k][from_node]){
            enqueue(&queue_list[k], from_node);
            flags[k][from_node] = true;
            }

            if(residue[k][to_node] > residuemax && !flags[k][to_node]){
            enqueue(&queue_list[k], to_node);
            flags[k][to_node] = true;
            }
        }
        }
    }
    }

}



















void DenseUndirected_Refresh_PPR_initialization_LP(int start, int end, UGraph* g, double residuemax, 
double alpha, vector<int>& labeled_node_vec, 
double** residue, 
double** pi,
bool** flags, 
Queue* queue_list,
int iter,
int vertex_number,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,


d_row_tree_mkl* subset_tree,

unordered_map<int, int> &row_index_mapping,
vector<int> &indicator,
int dynamic_ppr_start_iter
){

    if(iter < dynamic_ppr_start_iter){
      for(int i = start; i < end; i++){
        memset(residue[i], 0, sizeof(float) * vertex_number);
        memset(pi[i], 0, sizeof(float) * vertex_number);
        memset(flags[i], false, sizeof(bool) * vertex_number);
        queue_list[i].front = 0;
        queue_list[i].rear = 0;

        int src = labeled_node_vec[i];
        residue[i][src] = 1;
        flags[i][src] = true;
        enqueue(&queue_list[i], src);
      }
    }
    else{

    for(int i = 0; i < vertex_number; i++){
        int from_node = i;
        for(int j = g->former_degree[from_node]; j < g->degree[from_node]; j++){
        int to_node = g->AdjList[from_node][j];
        for(int k = start; k < end; k++){

            if(j == 0){
              continue;
            if(residue[k][from_node] / j > residuemax && !flags[k][from_node]){
                enqueue(&queue_list[k], from_node);
                flags[k][from_node] = true;
            }

            }

            

            int col_v = from_node / col_dim;
            
            if(col_v == nParts){
            col_v--;
            }


            int inner_col_index = inner_group_mapping[from_node];

            subset_tree->dense_mat_mapping[col_v](
            k, inner_col_index) += pi[k][from_node] * 1 / j;



            pi[k][from_node] *= (j + 1) / j;

            residue[k][from_node] -= pi[k][from_node] / (j+1) / alpha;
            residue[k][to_node] += (1 - alpha) * pi[k][from_node] / (j+1) / alpha;
            if(residue[k][from_node] / j > residuemax && !flags[k][from_node]){
            enqueue(&queue_list[k], from_node);
            flags[k][from_node] = true;
            }
            if(residue[k][to_node] / j > residuemax && !flags[k][to_node]){
            enqueue(&queue_list[k], to_node);
            flags[k][to_node] = true;
            }
        }
        }
    }
    }

}

















void Cache_pi_residue_to_parallel_blocks(
MatrixXf &residue,
MatrixXf &pi,
vector<SparseMatrix<float>> &sparse_pi_cache_vec,
vector<SparseMatrix<float>> &sparse_residue_cache_vec,
int count_labeled_node,
int start_column_number,
int block_column_size,
int block_number
){
  sparse_pi_cache_vec[block_number] = pi.block(0, start_column_number, count_labeled_node, block_column_size).sparseView();
  sparse_residue_cache_vec[block_number] = residue.block(0, start_column_number, count_labeled_node, block_column_size).sparseView();
}

void Retrieve_pi_residue_from_parallel_blocks(
MatrixXf &residue,
MatrixXf &pi,
vector<SparseMatrix<float>> &sparse_pi_cache_vec,
vector<SparseMatrix<float>> &sparse_residue_cache_vec,
int count_labeled_node,
int start_column_number,
int block_column_size,
int block_number
){


  for (int k_iter=0; k_iter<sparse_pi_cache_vec[block_number].outerSize(); ++k_iter){
      for (SparseMatrix<float, ColMajor, int>::InnerIterator it(sparse_pi_cache_vec[block_number], k_iter); it; ++it){
          pi(it.row(), it.col()) = it.value();
      }
  }

  for (int k_iter=0; k_iter<sparse_residue_cache_vec[block_number].outerSize(); ++k_iter){
      for (SparseMatrix<float, ColMajor, int>::InnerIterator it(sparse_residue_cache_vec[block_number], k_iter); it; ++it){
          residue(it.row(), it.col()) = it.value();
      }
  }

}


void Cache_pi_residue_transpose_to_parallel_blocks(
MatrixXf &residue_transpose,
MatrixXf &pi_transpose,
vector<SparseMatrix<float>> &sparse_residue_transpose_cache_vec,
vector<SparseMatrix<float>> &sparse_pi_transpose_cache_vec,
int count_labeled_node,
int start_column_number,
int block_column_size,
int block_number
){

  sparse_pi_transpose_cache_vec[block_number] 
    = pi_transpose.block(0, start_column_number, count_labeled_node, block_column_size).sparseView();
  sparse_residue_transpose_cache_vec[block_number] 
    = residue_transpose.block(0, start_column_number, count_labeled_node, block_column_size).sparseView();

}

void Retrieve_pi_residue_transpose_from_parallel_blocks(
MatrixXf &residue_transpose,
MatrixXf &pi_transpose,
vector<SparseMatrix<float>> &sparse_residue_transpose_cache_vec,
vector<SparseMatrix<float>> &sparse_pi_transpose_cache_vec,
int count_labeled_node,
int start_column_number,
int block_column_size,
int block_number
){

  for (int k_iter=0; k_iter<sparse_pi_transpose_cache_vec[block_number].outerSize(); ++k_iter){
      for (SparseMatrix<float, ColMajor, int>::InnerIterator it(sparse_pi_transpose_cache_vec[block_number], k_iter); it; ++it){
          pi_transpose(it.row(), it.col()) = it.value();
      }
  }

  for (int k_iter=0; k_iter<sparse_residue_transpose_cache_vec[block_number].outerSize(); ++k_iter){
      for (SparseMatrix<float, ColMajor, int>::InnerIterator it(sparse_residue_transpose_cache_vec[block_number], k_iter); it; ++it){
          residue_transpose(it.row(), it.col()) = it.value();
      }
  }


}









SparseMatrix<double> sparseBlock(SparseMatrix<double, ColMajor, int64_t> M,
        int ibegin, int jbegin, int icount, int jcount){
        //only for ColMajor Sparse Matrix
        
    typedef Triplet<double> Tri;
    
    
    assert(ibegin+icount <= M.rows());
    assert(jbegin+jcount <= M.cols());
    int Mj,Mi,i,j,currOuterIndex,nextOuterIndex;
    vector<Tri> tripletList;
    tripletList.reserve(M.nonZeros());

    for(j=0; j<jcount; j++){
        Mj=j+jbegin;
        currOuterIndex = M.outerIndexPtr()[Mj];
        nextOuterIndex = M.outerIndexPtr()[Mj+1];

        for(int a = currOuterIndex; a<nextOuterIndex; a++){
            Mi=M.innerIndexPtr()[a];

            if(Mi < ibegin) continue;
            if(Mi >= ibegin + icount) break;

            i=Mi-ibegin;    
            tripletList.push_back(Tri(i,j,M.valuePtr()[a]));
        }
    }
    SparseMatrix<double> matS(icount,jcount);
    matS.setFromTriplets(tripletList.begin(), tripletList.end());
    return matS;
}


void Log_sparse_matrix_entries_sparse_tree(
int k, int i,    
double reservemin, 
vector<sparse_d_row_tree_mkl*> &d_row_tree_vec,
unordered_map<int, vector<int>> &vec_mapping,
SparseMatrix<double, 0, int64_t> &subset_trMat
){

    SparseMatrix<double, 0, int64_t> &current_mat_mapping = d_row_tree_vec[k]->mat_mapping[i];


    current_mat_mapping.resize(0, 0);


    int temp_row_dim = d_row_tree_vec[k]->row_dim;
    int current_group_size = vec_mapping[i].size();
    current_mat_mapping.resize(subset_trMat.rows(), current_group_size);



    if(i != vec_mapping.size() - 1){
      int start_col_index = i * vec_mapping[i].size();
      current_mat_mapping = sparseBlock(subset_trMat, 0, start_col_index, subset_trMat.rows(),  vec_mapping[i].size());

    }
    else{
      int start_col_index = i * vec_mapping[i-1].size();
      current_mat_mapping = sparseBlock(subset_trMat, 0, start_col_index, subset_trMat.rows(),  vec_mapping[i].size());

    }



}











void Log_sparse_matrix_entries(
int i,    
double reservemin, 

d_row_tree_mkl* subset_tree,
unordered_map<int, vector<int>> &vec_mapping,
vector<long long int>& record_submatrices_nnz
){

    SparseMatrix<double, 0, int> &current_mat_mapping = subset_tree->mat_mapping[i];

    current_mat_mapping.resize(0, 0);


    record_submatrices_nnz[i] = 0;

    int temp_row_dim = subset_tree->row_dim;
    int current_group_size = vec_mapping[i].size();
    current_mat_mapping.resize(temp_row_dim, current_group_size);
    current_mat_mapping = subset_tree->dense_mat_mapping[i].sparseView();
    for (int k_iter=0; k_iter<subset_tree->mat_mapping[i].outerSize(); ++k_iter){
        for (SparseMatrix<double, ColMajor, int>::InnerIterator it(subset_tree->mat_mapping[i], k_iter); it; ++it){
            if(it.value() > reservemin){
                it.valueRef() = log10(it.value()/reservemin);
                record_submatrices_nnz[i]++;
            }
            else{
                it.valueRef() = 0;
            }


        }
    }

}






void No_Log_sparse_matrix_entries_LP(
int i,    
double reservemin, 

d_row_tree_mkl* subset_tree,
unordered_map<int, vector<int>> &vec_mapping,
vector<long long int>& record_submatrices_nnz
){

    SparseMatrix<double, 0, int> &current_mat_mapping = subset_tree->mat_mapping[i];

    current_mat_mapping.resize(0, 0);

    int temp_row_dim = subset_tree->row_dim;
    int current_group_size = vec_mapping[i].size();
    current_mat_mapping.resize(temp_row_dim, current_group_size);
    current_mat_mapping = subset_tree->dense_mat_mapping[i].sparseView();
    for (int k_iter=0; k_iter<subset_tree->mat_mapping[i].outerSize(); ++k_iter){
        for (SparseMatrix<double, ColMajor, int>::InnerIterator it(subset_tree->mat_mapping[i], k_iter); it; ++it){

            it.valueRef() = it.value();
            record_submatrices_nnz[i]++;

        }
    }

}



void Log_sparse_matrix_entries_LP(
int i,    
double reservemin, 
d_row_tree_mkl* subset_tree,
unordered_map<int, vector<int>> &vec_mapping,
vector<long long int>& record_submatrices_nnz
){

    SparseMatrix<double, 0, int> &current_mat_mapping = subset_tree->mat_mapping[i];

    current_mat_mapping.resize(0, 0);




    record_submatrices_nnz[i] = 0;




    int temp_row_dim = subset_tree->row_dim;
    int current_group_size = vec_mapping[i].size();
    current_mat_mapping.resize(temp_row_dim, current_group_size);
    current_mat_mapping = subset_tree->dense_mat_mapping[i].sparseView();
    for (int k_iter=0; k_iter<subset_tree->mat_mapping[i].outerSize(); ++k_iter){
        for (SparseMatrix<double, ColMajor, int>::InnerIterator it(subset_tree->mat_mapping[i], k_iter); it; ++it){
            if(it.value() > reservemin){
                it.valueRef() = log10(1 + it.value()/reservemin);
                record_submatrices_nnz[i]++;
            }
            else{

                it.valueRef() = log10(1 + it.value()/reservemin);
                record_submatrices_nnz[i]++;
            }
            

        }
    }

}










void Log_sparse_matrix_entries_with_norm_computation(
// int k, 
int i,    
double reservemin, 

d_row_tree_mkl* subset_tree,
unordered_map<int, vector<int>> &vec_mapping,

vector<int>& update_mat_tree_record,
int iter,
double delta,
int count_labeled_node,
int d,
vector<long long int>& record_submatrices_nnz
){
    
    SparseMatrix<double, 0, int> &old_mat_mapping = subset_tree->mat_mapping[i];

    int temp_row_dim = subset_tree->row_dim;
    int current_group_size = vec_mapping[i].size();


    SparseMatrix<double, 0, int> current_mat_mapping;


    current_mat_mapping.resize(temp_row_dim, current_group_size);
    current_mat_mapping = subset_tree->dense_mat_mapping[i].sparseView();


    long long int temp_record_submatrices_nnz = 0;

    for (int k_iter=0; k_iter<current_mat_mapping.outerSize(); ++k_iter){
        for (SparseMatrix<double, ColMajor, int>::InnerIterator it(current_mat_mapping, k_iter); it; ++it){
            if(it.value() > reservemin){
                it.valueRef() = log10(it.value()/reservemin);

                temp_record_submatrices_nnz++;
            }
            else if(it.value() == 0){

            }
            else{
                it.valueRef() = 0;
            }

        }
    }



    double A_norm = current_mat_mapping.norm();



    double Ei_norm = (current_mat_mapping - old_mat_mapping).norm();



    delta = delta * sqrt(2);



    if( subset_tree->norm_B_Bid_difference_vec[i] + Ei_norm < delta * A_norm){
      update_mat_tree_record[i] = -1;
      current_mat_mapping.resize(0, 0);
      current_mat_mapping.data().squeeze();


    }
    else{
      update_mat_tree_record[i] = iter;
      old_mat_mapping.resize(0, 0);
      old_mat_mapping.data().squeeze();
      subset_tree->mat_mapping[i] = current_mat_mapping;

      record_submatrices_nnz[i] = temp_record_submatrices_nnz;


    }




}









void Log_sparse_matrix_entries_with_norm_computation_LP(
int i,    
double reservemin, 

d_row_tree_mkl* subset_tree,
unordered_map<int, vector<int>> &vec_mapping,

vector<int>& update_mat_tree_record,
int iter,
double delta,
int count_labeled_node,
int d,
vector<long long int>& record_submatrices_nnz
){


    
    SparseMatrix<double, 0, int> &old_mat_mapping = subset_tree->mat_mapping[i];

    int temp_row_dim = subset_tree->row_dim;
    int current_group_size = vec_mapping[i].size();



    SparseMatrix<double, 0, int> current_mat_mapping;


    current_mat_mapping.resize(temp_row_dim, current_group_size);
    current_mat_mapping = subset_tree->dense_mat_mapping[i].sparseView();


    long long int temp_record_submatrices_nnz = 0;

    for (int k_iter=0; k_iter<current_mat_mapping.outerSize(); ++k_iter){
        for (SparseMatrix<double, ColMajor, int>::InnerIterator it(current_mat_mapping, k_iter); it; ++it){
            if(it.value() > reservemin){
                it.valueRef() = log10(1 + it.value()/reservemin);

                temp_record_submatrices_nnz++;
            }
            else if(it.value() == 0){

            }
            else{
                it.valueRef() = log10(1 + it.value()/reservemin);
                temp_record_submatrices_nnz++;
            }

        }
    }



    double A_norm = current_mat_mapping.norm();



    double Ei_norm = (current_mat_mapping - old_mat_mapping).norm();



    delta = delta * sqrt(2);



    if( subset_tree->norm_B_Bid_difference_vec[i] + Ei_norm < delta * A_norm){
      update_mat_tree_record[i] = -1;
      current_mat_mapping.resize(0, 0);
      current_mat_mapping.data().squeeze();

    }
    else{
      update_mat_tree_record[i] = iter;
      old_mat_mapping.resize(0, 0);
      old_mat_mapping.data().squeeze();
      subset_tree->mat_mapping[i] = current_mat_mapping;

      record_submatrices_nnz[i] = temp_record_submatrices_nnz;

    }




}





















void DenseDirectedDynamicForwardPush(int start, int end, Graph* g, double residuemax, double reservemin, 
double alpha, vector<int>& labeled_node_vec, 
float** residue, 
float** pi,
bool** flags, 
Queue* queue_list,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,

d_row_tree_mkl * subset_tree
)
{

  int vertices_number = g->n;  

  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);

      if(g->outdegree[v] == 0){
        flags[it][v] = false;
        queue_list[it].front++;

        continue;
      }



      if(residue[it][v] / g->outdegree[v] > residuemax){
        for(int j = 0; j < g->outdegree[v]; j++){
          int u = g->outAdjList[v][j];
          residue[it][u] += (1-alpha) * residue[it][v] / g->outdegree[v];
          
          if(g->outdegree[u] == 0){
            continue;
          }
          
          if(residue[it][u] / g->outdegree[u] > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }
        }
        pi[it][v] += alpha * residue[it][v];

        int col_v = v / col_dim;


        if(col_v == nParts){
          col_v--;
        }

        int inner_col_index = inner_group_mapping[v];


        subset_tree->dense_mat_mapping[col_v](
          it, inner_col_index) += alpha * residue[it][v];

        residue[it][v] = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }


    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }

  return;

}















void nodegree_DenseDirectedDynamicForwardPush(int start, int end, Graph* g, double residuemax, double reservemin, 
double alpha, vector<int>& labeled_node_vec, 
float** residue, 
float** pi,
bool** flags, 
Queue* queue_list,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,

d_row_tree_mkl * subset_tree
)
{

  int vertices_number = g->n;  

  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);

      if(g->outdegree[v] == 0){
        flags[it][v] = false;
        queue_list[it].front++;

        continue;
      }




      if(residue[it][v] > residuemax){
        for(int j = 0; j < g->outdegree[v]; j++){
          int u = g->outAdjList[v][j];
          residue[it][u] += (1-alpha) * residue[it][v] / g->outdegree[v];
          
          if(g->outdegree[u] == 0){

            continue;
          }
          

          if(residue[it][u] > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }
        }
        pi[it][v] += alpha * residue[it][v];


        int col_v = v / col_dim;


        if(col_v == nParts){
          col_v--;
        }

        int inner_col_index = inner_group_mapping[v];


        subset_tree->dense_mat_mapping[col_v](
          it, inner_col_index) += alpha * residue[it][v];

        residue[it][v] = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }


    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }

  return;

}














void nodegree_DenseDirectedDynamicForwardPush_LP(int start, int end, Graph* g, double residuemax, double reservemin, 
double alpha, vector<int>& labeled_node_vec, 
double** residue, 
double** pi,
bool** flags, 
Queue* queue_list,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,

d_row_tree_mkl * subset_tree
)
{

  int vertices_number = g->n;  

  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);

      if(g->outdegree[v] == 0){
        flags[it][v] = false;
        queue_list[it].front++;

        continue;
      }




      if(residue[it][v] > residuemax){
        for(int j = 0; j < g->outdegree[v]; j++){
          int u = g->outAdjList[v][j];
          residue[it][u] += (1-alpha) * residue[it][v] / g->outdegree[v];
          
          if(g->outdegree[u] == 0){

            continue;
          }
          

          if(residue[it][u] > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }
        }
        pi[it][v] += alpha * residue[it][v];


        int col_v = v / col_dim;
        
        if(col_v == nParts){
          col_v--;
        }

        int inner_col_index = inner_group_mapping[v];


        subset_tree->dense_mat_mapping[col_v](
          it, inner_col_index) += alpha * residue[it][v];

        residue[it][v] = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }


    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }

  return;

}






















void DenseDirectedDynamicForwardPush_LP(int start, int end, Graph* g, double residuemax, double reservemin, 
double alpha, vector<int>& labeled_node_vec, 
double** residue, 
double** pi,
bool** flags, 
Queue* queue_list,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,

d_row_tree_mkl * subset_tree
)
{

  int vertices_number = g->n;  

  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);

      if(g->outdegree[v] == 0){
        flags[it][v] = false;
        queue_list[it].front++;

        continue;
      }



      if(residue[it][v] / g->outdegree[v] > residuemax){
        for(int j = 0; j < g->outdegree[v]; j++){
          int u = g->outAdjList[v][j];
          residue[it][u] += (1-alpha) * residue[it][v] / g->outdegree[v];
          
          if(g->outdegree[u] == 0){

            continue;
          }
          
          if(residue[it][u] / g->outdegree[u] > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }
        }
        pi[it][v] += alpha * residue[it][v];


        int col_v = v / col_dim;
        
        if(col_v == nParts){
          col_v--;
        }

        int inner_col_index = inner_group_mapping[v];


        subset_tree->dense_mat_mapping[col_v](
          it, inner_col_index) += alpha * residue[it][v];

        residue[it][v] = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }


    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }

  return;

}
















void DenseDirectedDynamicForwardPushTranspose(int start, int end, Graph* g, double residuemax, double reservemin, 
double alpha, vector<int>& labeled_node_vec, 
float** residue, 
float** pi,
bool** flags, 
Queue* queue_list,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,
d_row_tree_mkl* subset_tree
)
{

  int vertices_number = g->n;  

  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);

      if(g->indegree[v] == 0){
        flags[it][v] = false;
        queue_list[it].front++;

        continue;
      }

      if(residue[it][v] / g->indegree[v] > residuemax){
        for(int j = 0; j < g->indegree[v]; j++){
          int u = g->inAdjList[v][j];
          residue[it][u] += (1-alpha) * residue[it][v] / g->indegree[v];

          if(g->indegree[u] == 0){

            continue;
          }
          
          if(residue[it][u] / g->indegree[u] > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }
        }
        pi[it][v] += alpha * residue[it][v];


        int col_v = v / col_dim;
        
        if(col_v == nParts){
          col_v--;
        }

        int inner_col_index = inner_group_mapping[v];


        subset_tree->dense_mat_mapping[col_v](
          it, inner_col_index) += alpha * residue[it][v];

        residue[it][v] = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }

    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }

  return;

}


























void nodegree_DenseDirectedDynamicForwardPushTranspose(int start, int end, Graph* g, double residuemax, double reservemin, 
double alpha, vector<int>& labeled_node_vec, 
float** residue, 
float** pi,
bool** flags, 
Queue* queue_list,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,
d_row_tree_mkl* subset_tree

)
{

  int vertices_number = g->n;  

  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);

      if(g->indegree[v] == 0){
        flags[it][v] = false;
        queue_list[it].front++;

        continue;
      }


      if(residue[it][v] > residuemax){
        for(int j = 0; j < g->indegree[v]; j++){
          int u = g->inAdjList[v][j];
          residue[it][u] += (1-alpha) * residue[it][v] / g->indegree[v];

          if(g->indegree[u] == 0){

            continue;
          }
          

          if(residue[it][u] > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }
        }
        pi[it][v] += alpha * residue[it][v];


        int col_v = v / col_dim;
        
        if(col_v == nParts){
          col_v--;
        }

        int inner_col_index = inner_group_mapping[v];


        subset_tree->dense_mat_mapping[col_v](
          it, inner_col_index) += alpha * residue[it][v];

        residue[it][v] = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }

    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }

  return;

}





















void nodegree_DenseDirectedDynamicForwardPushTranspose_LP(int start, int end, Graph* g, double residuemax, double reservemin, 
double alpha, vector<int>& labeled_node_vec, 
double** residue, 
double** pi,
bool** flags, 
Queue* queue_list,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,
d_row_tree_mkl* subset_tree

)
{

  int vertices_number = g->n;  

  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);

      if(g->indegree[v] == 0){
        flags[it][v] = false;
        queue_list[it].front++;

        continue;
      }


      if(residue[it][v] > residuemax){
        for(int j = 0; j < g->indegree[v]; j++){
          int u = g->inAdjList[v][j];
          residue[it][u] += (1-alpha) * residue[it][v] / g->indegree[v];

          if(g->indegree[u] == 0){

            continue;
          }
          

          if(residue[it][u] > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }
        }
        pi[it][v] += alpha * residue[it][v];


        int col_v = v / col_dim;
        
        if(col_v == nParts){
          col_v--;
        }

        int inner_col_index = inner_group_mapping[v];


        subset_tree->dense_mat_mapping[col_v](
          it, inner_col_index) += alpha * residue[it][v];

        residue[it][v] = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }

    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }

  return;

}





































void DenseDirectedDynamicForwardPushTranspose_LP(int start, int end, Graph* g, double residuemax, double reservemin, 
double alpha, vector<int>& labeled_node_vec, 
double** residue, 
double** pi,
bool** flags, 
Queue* queue_list,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,
d_row_tree_mkl* subset_tree

)
{

  int vertices_number = g->n;  

  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);

      if(g->indegree[v] == 0){
        flags[it][v] = false;
        queue_list[it].front++;

        continue;
      }

      if(residue[it][v] / g->indegree[v] > residuemax){
        for(int j = 0; j < g->indegree[v]; j++){
          int u = g->inAdjList[v][j];
          residue[it][u] += (1-alpha) * residue[it][v] / g->indegree[v];

          if(g->indegree[u] == 0){

            continue;
          }
          
          if(residue[it][u] / g->indegree[u] > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }
        }
        pi[it][v] += alpha * residue[it][v];


        int col_v = v / col_dim;
        
        if(col_v == nParts){
          col_v--;
        }

        int inner_col_index = inner_group_mapping[v];


        subset_tree->dense_mat_mapping[col_v](
          it, inner_col_index) += alpha * residue[it][v];

        residue[it][v] = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }

    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }

  return;

}








































void DenseDirected_Refresh_PPR_initialization(int start, int end, Graph* g, double residuemax, double reservemin,
double alpha, vector<int>& labeled_node_vec, 

float** residue, 
float** pi,
bool** flags, 
Queue* queue_list,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,

int iter,
int vertex_number,

d_row_tree_mkl* subset_tree,
int dynamic_ppr_start_iter
){

  if(iter < dynamic_ppr_start_iter){
    for(int i = start; i < end; i++){
      memset(residue[i], 0, sizeof(float) * vertex_number);
      memset(pi[i], 0, sizeof(float) * vertex_number);
      memset(flags[i], false, sizeof(bool) * vertex_number);
      queue_list[i].front = 0;
      queue_list[i].rear = 0;

      int src = labeled_node_vec[i];
      residue[i][src] = 1;
      flags[i][src] = true;
      enqueue(&queue_list[i], src);
    }
  }
  else{

    for(int i = 0; i < vertex_number; i++){
      int from_node = i;
      for(int j = g->former_outdegree[from_node]; j < g->outdegree[from_node]; j++){
        int to_node = g->outAdjList[from_node][j];
        
        for(int k = start; k < end; k++){
          if(j == 0){
            continue;
            if(residue[k][from_node] / j > residuemax && !flags[k][from_node]){
              enqueue(&queue_list[k], from_node);
              flags[k][from_node] = true;
            }
          }


          int col_v = from_node / col_dim;
          
          if(col_v == nParts){
            col_v--;
          }

          int inner_col_index = inner_group_mapping[from_node];

          subset_tree->dense_mat_mapping[col_v](
            k , inner_col_index) += pi[k][from_node] * 1 / j;

          pi[k][from_node] *= (j + 1) / j;

          residue[k][from_node] -= pi[k][from_node] / (j+1) / alpha;
          residue[k][to_node] += (1 - alpha) * pi[k][from_node] / (j+1) / alpha;
          if(residue[k][from_node] / j > residuemax && !flags[k][from_node]){
            enqueue(&queue_list[k], from_node);
            flags[k][from_node] = true;
          }
          if(residue[k][to_node] / j > residuemax && !flags[k][to_node]){
            enqueue(&queue_list[k], to_node);
            flags[k][to_node] = true;
          }
        }

      }
    }
  }




}





























void nodegree_DenseDirected_Refresh_PPR_initialization(int start, int end, Graph* g, double residuemax, double reservemin,
double alpha, vector<int>& labeled_node_vec, 

float** residue, 
float** pi,
bool** flags, 
Queue* queue_list,


int col_dim,

int nParts,
vector<int> &inner_group_mapping,

int iter,
int vertex_number,

d_row_tree_mkl* subset_tree,
int dynamic_ppr_start_iter
){

  if(iter < dynamic_ppr_start_iter){
    for(int i = start; i < end; i++){
      memset(residue[i], 0, sizeof(float) * vertex_number);
      memset(pi[i], 0, sizeof(float) * vertex_number);
      memset(flags[i], false, sizeof(bool) * vertex_number);
      queue_list[i].front = 0;
      queue_list[i].rear = 0;

      int src = labeled_node_vec[i];
      residue[i][src] = 1;
      flags[i][src] = true;
      enqueue(&queue_list[i], src);
    }
  }
  else{

    for(int i = 0; i < vertex_number; i++){
      int from_node = i;
      for(int j = g->former_outdegree[from_node]; j < g->outdegree[from_node]; j++){
        int to_node = g->outAdjList[from_node][j];
        
        for(int k = start; k < end; k++){
          if(j == 0){
            continue;

            if(residue[k][from_node] > residuemax && !flags[k][from_node]){
              enqueue(&queue_list[k], from_node);
              flags[k][from_node] = true;
            }
          }


          int col_v = from_node / col_dim;
          
          if(col_v == nParts){
            col_v--;
          }

          int inner_col_index = inner_group_mapping[from_node];

          subset_tree->dense_mat_mapping[col_v](
            k , inner_col_index) += pi[k][from_node] * 1 / j;

          pi[k][from_node] *= (j + 1) / j;

          residue[k][from_node] -= pi[k][from_node] / (j+1) / alpha;
          residue[k][to_node] += (1 - alpha) * pi[k][from_node] / (j+1) / alpha;

          if(residue[k][from_node] > residuemax && !flags[k][from_node]){
            enqueue(&queue_list[k], from_node);
            flags[k][from_node] = true;
          }

          if(residue[k][to_node] > residuemax && !flags[k][to_node]){
            enqueue(&queue_list[k], to_node);
            flags[k][to_node] = true;
          }
        }

      }
    }
  }




}










void nodegree_DenseDirected_Refresh_PPR_initialization_LP(int start, int end, Graph* g, double residuemax, double reservemin,
double alpha, vector<int>& labeled_node_vec, 

double** residue, 
double** pi,
bool** flags, 
Queue* queue_list,



int col_dim,

int nParts,
vector<int> &inner_group_mapping,

int iter,
int vertex_number,

d_row_tree_mkl* subset_tree,
int dynamic_ppr_start_iter
){

  if(iter < dynamic_ppr_start_iter){
    for(int i = start; i < end; i++){
      memset(residue[i], 0, sizeof(float) * vertex_number);
      memset(pi[i], 0, sizeof(float) * vertex_number);
      memset(flags[i], false, sizeof(bool) * vertex_number);
      queue_list[i].front = 0;
      queue_list[i].rear = 0;

      int src = labeled_node_vec[i];
      residue[i][src] = 1;
      flags[i][src] = true;
      enqueue(&queue_list[i], src);
    }
  }
  else{

    for(int i = 0; i < vertex_number; i++){
      int from_node = i;
      for(int j = g->former_outdegree[from_node]; j < g->outdegree[from_node]; j++){
        int to_node = g->outAdjList[from_node][j];
        
        for(int k = start; k < end; k++){
          if(j == 0){
            continue;

            if(residue[k][from_node] > residuemax && !flags[k][from_node]){
              enqueue(&queue_list[k], from_node);
              flags[k][from_node] = true;
            }
          }


          int col_v = from_node / col_dim;
          
          if(col_v == nParts){
            col_v--;
          }

          int inner_col_index = inner_group_mapping[from_node];

          subset_tree->dense_mat_mapping[col_v](
            k , inner_col_index) += pi[k][from_node] * 1 / j;

          pi[k][from_node] *= (j + 1) / j;

          residue[k][from_node] -= pi[k][from_node] / (j+1) / alpha;
          residue[k][to_node] += (1 - alpha) * pi[k][from_node] / (j+1) / alpha;

          if(residue[k][from_node] > residuemax && !flags[k][from_node]){
            enqueue(&queue_list[k], from_node);
            flags[k][from_node] = true;
          }

          if(residue[k][to_node] > residuemax && !flags[k][to_node]){
            enqueue(&queue_list[k], to_node);
            flags[k][to_node] = true;
          }
        }

      }
    }
  }




}


























void DenseDirected_Refresh_PPR_initialization_Transpose(int start, int end, Graph* g, double residuemax, double reservemin,
double alpha, vector<int>& labeled_node_vec, 

float** residue_transpose, 
float** pi_transpose,
bool** flags_transpose, 
Queue * queue_list_transpose,


int col_dim,

int nParts,
vector<int> &inner_group_mapping,

int iter,
int vertex_number,

d_row_tree_mkl* subset_tree,
int dynamic_ppr_start_iter
){




  if(iter < dynamic_ppr_start_iter){
    for(int i = start; i < end; i++){
      memset(residue_transpose[i], 0, sizeof(float) * vertex_number);
      memset(pi_transpose[i], 0, sizeof(float) * vertex_number);
      memset(flags_transpose[i], false, sizeof(bool) * vertex_number);

      queue_list_transpose[i].front = 0;
      queue_list_transpose[i].rear = 0;

      int src = labeled_node_vec[i];
      residue_transpose[i][src] = 1;
      flags_transpose[i][src] = true;
      enqueue(&queue_list_transpose[i], src);
    }

  }
  else{

    for(int i = 0; i < vertex_number; i++){
      int from_node = i;
      for(int j = g->former_indegree[from_node]; j < g->indegree[from_node]; j++){
        int to_node = g->inAdjList[from_node][j];

        for(int k = start; k < end; k++){

          if(j == 0){
            continue;
            if(residue_transpose[k][from_node] / j > residuemax && !flags_transpose[k][from_node]){
              enqueue(&queue_list_transpose[k], from_node);
              flags_transpose[k][from_node] = true;
            }

          }


          int col_v = from_node / col_dim;
          
          if(col_v == nParts){
            col_v--;
          }

          int inner_col_index = inner_group_mapping[from_node];

          subset_tree->dense_mat_mapping[col_v](
            k, inner_col_index) += pi_transpose[k][from_node] * 1 / j;

          pi_transpose[k][from_node] *= (j + 1) / j;

          residue_transpose[k][from_node] -= pi_transpose[k][from_node] / (j+1) / alpha;
          residue_transpose[k][to_node] += (1 - alpha) * pi_transpose[k][from_node] / (j+1) / alpha;
          if(residue_transpose[k][from_node] / j > residuemax && !flags_transpose[k][from_node]){
            enqueue(&queue_list_transpose[k], from_node);
            flags_transpose[k][from_node] = true;
          }
          if(residue_transpose[k][to_node] / j > residuemax && !flags_transpose[k][to_node]){
            enqueue(&queue_list_transpose[k], to_node);
            flags_transpose[k][to_node] = true;
          }
        }
        
      }
    }
  }






}















void nodegree_DenseDirected_Refresh_PPR_initialization_Transpose(int start, int end, Graph* g, double residuemax, double reservemin,
double alpha, vector<int>& labeled_node_vec, 

float** residue_transpose, 
float** pi_transpose,
bool** flags_transpose, 
Queue * queue_list_transpose,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,

int iter,
int vertex_number,

d_row_tree_mkl* subset_tree,
int dynamic_ppr_start_iter
){




  if(iter < dynamic_ppr_start_iter){
    for(int i = start; i < end; i++){
      memset(residue_transpose[i], 0, sizeof(float) * vertex_number);
      memset(pi_transpose[i], 0, sizeof(float) * vertex_number);
      memset(flags_transpose[i], false, sizeof(bool) * vertex_number);

      queue_list_transpose[i].front = 0;
      queue_list_transpose[i].rear = 0;

      int src = labeled_node_vec[i];
      residue_transpose[i][src] = 1;
      flags_transpose[i][src] = true;
      enqueue(&queue_list_transpose[i], src);
    }

  }
  else{

    for(int i = 0; i < vertex_number; i++){
      int from_node = i;
      for(int j = g->former_indegree[from_node]; j < g->indegree[from_node]; j++){
        int to_node = g->inAdjList[from_node][j];

        for(int k = start; k < end; k++){

          if(j == 0){
            continue;

            if(residue_transpose[k][from_node] > residuemax && !flags_transpose[k][from_node]){
              enqueue(&queue_list_transpose[k], from_node);
              flags_transpose[k][from_node] = true;
            }

          }


          int col_v = from_node / col_dim;
          
          if(col_v == nParts){
            col_v--;
          }

          int inner_col_index = inner_group_mapping[from_node];

          subset_tree->dense_mat_mapping[col_v](
            k, inner_col_index) += pi_transpose[k][from_node] * 1 / j;

          pi_transpose[k][from_node] *= (j + 1) / j;

          residue_transpose[k][from_node] -= pi_transpose[k][from_node] / (j+1) / alpha;
          residue_transpose[k][to_node] += (1 - alpha) * pi_transpose[k][from_node] / (j+1) / alpha;

          if(residue_transpose[k][from_node] > residuemax && !flags_transpose[k][from_node]){
            enqueue(&queue_list_transpose[k], from_node);
            flags_transpose[k][from_node] = true;
          }

          if(residue_transpose[k][to_node] > residuemax && !flags_transpose[k][to_node]){
            enqueue(&queue_list_transpose[k], to_node);
            flags_transpose[k][to_node] = true;
          }
        }
        
      }
    }
  }






}















void nodegree_DenseDirected_Refresh_PPR_initialization_Transpose_LP(int start, int end, Graph* g, double residuemax, double reservemin,
double alpha, vector<int>& labeled_node_vec, 

double** residue_transpose, 
double** pi_transpose,
bool** flags_transpose, 
Queue * queue_list_transpose,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,

int iter,
int vertex_number,

d_row_tree_mkl* subset_tree,
int dynamic_ppr_start_iter
){




  if(iter < dynamic_ppr_start_iter){
    for(int i = start; i < end; i++){
      memset(residue_transpose[i], 0, sizeof(float) * vertex_number);
      memset(pi_transpose[i], 0, sizeof(float) * vertex_number);
      memset(flags_transpose[i], false, sizeof(bool) * vertex_number);

      queue_list_transpose[i].front = 0;
      queue_list_transpose[i].rear = 0;

      int src = labeled_node_vec[i];
      residue_transpose[i][src] = 1;
      flags_transpose[i][src] = true;
      enqueue(&queue_list_transpose[i], src);
    }

  }
  else{

    for(int i = 0; i < vertex_number; i++){
      int from_node = i;
      for(int j = g->former_indegree[from_node]; j < g->indegree[from_node]; j++){
        int to_node = g->inAdjList[from_node][j];

        for(int k = start; k < end; k++){

          if(j == 0){
            continue;

            if(residue_transpose[k][from_node] > residuemax && !flags_transpose[k][from_node]){
              enqueue(&queue_list_transpose[k], from_node);
              flags_transpose[k][from_node] = true;
            }

          }


          int col_v = from_node / col_dim;
          
          if(col_v == nParts){
            col_v--;
          }

          int inner_col_index = inner_group_mapping[from_node];

          subset_tree->dense_mat_mapping[col_v](
            k, inner_col_index) += pi_transpose[k][from_node] * 1 / j;

          pi_transpose[k][from_node] *= (j + 1) / j;

          residue_transpose[k][from_node] -= pi_transpose[k][from_node] / (j+1) / alpha;
          residue_transpose[k][to_node] += (1 - alpha) * pi_transpose[k][from_node] / (j+1) / alpha;

          if(residue_transpose[k][from_node] > residuemax && !flags_transpose[k][from_node]){
            enqueue(&queue_list_transpose[k], from_node);
            flags_transpose[k][from_node] = true;
          }

          if(residue_transpose[k][to_node] > residuemax && !flags_transpose[k][to_node]){
            enqueue(&queue_list_transpose[k], to_node);
            flags_transpose[k][to_node] = true;
          }
        }
        
      }
    }
  }






}

























void DenseDirected_Refresh_PPR_initialization_LP(int start, int end, Graph* g, double residuemax, double reservemin,
double alpha, vector<int>& labeled_node_vec, 

double** residue, 
double** pi,
bool** flags, 
Queue* queue_list,

double** residue_transpose, 
double** pi_transpose,
bool** flags_transpose, 
Queue * queue_list_transpose,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,

int iter,
int vertex_number,

d_row_tree_mkl* subset_tree,
int dynamic_ppr_start_iter
){

  if(iter < dynamic_ppr_start_iter){
    for(int i = start; i < end; i++){
      memset(residue[i], 0, sizeof(float) * vertex_number);
      memset(pi[i], 0, sizeof(float) * vertex_number);
      memset(flags[i], false, sizeof(bool) * vertex_number);
      queue_list[i].front = 0;
      queue_list[i].rear = 0;

      int src = labeled_node_vec[i];
      residue[i][src] = 1;
      flags[i][src] = true;
      enqueue(&queue_list[i], src);
    }
  }
  else{

    for(int i = 0; i < vertex_number; i++){
      int from_node = i;
      for(int j = g->former_outdegree[from_node]; j < g->outdegree[from_node]; j++){
        int to_node = g->outAdjList[from_node][j];
        
        for(int k = start; k < end; k++){
          if(j == 0){
            continue;
            if(residue[k][from_node] / j > residuemax && !flags[k][from_node]){
              enqueue(&queue_list[k], from_node);
              flags[k][from_node] = true;
            }
          }


          int col_v = from_node / col_dim;
          
          if(col_v == nParts){
            col_v--;
          }

          int inner_col_index = inner_group_mapping[from_node];

          subset_tree->dense_mat_mapping[col_v](
            k , inner_col_index) += pi[k][from_node] * 1 / j;

          pi[k][from_node] *= (j + 1) / j;

          residue[k][from_node] -= pi[k][from_node] / (j+1) / alpha;
          residue[k][to_node] += (1 - alpha) * pi[k][from_node] / (j+1) / alpha;
          if(residue[k][from_node] / j > residuemax && !flags[k][from_node]){
            enqueue(&queue_list[k], from_node);
            flags[k][from_node] = true;
          }
          if(residue[k][to_node] / j > residuemax && !flags[k][to_node]){
            enqueue(&queue_list[k], to_node);
            flags[k][to_node] = true;
          }
        }

      }
    }
  }


  if(iter < dynamic_ppr_start_iter){
    for(int i = start; i < end; i++){
      memset(residue_transpose[i], 0, sizeof(float) * vertex_number);
      memset(pi_transpose[i], 0, sizeof(float) * vertex_number);
      memset(flags_transpose[i], false, sizeof(bool) * vertex_number);

      queue_list_transpose[i].front = 0;
      queue_list_transpose[i].rear = 0;

      int src = labeled_node_vec[i];
      residue_transpose[i][src] = 1;
      flags_transpose[i][src] = true;
      enqueue(&queue_list_transpose[i], src);
    }

  }
  else{

    for(int i = 0; i < vertex_number; i++){
      int from_node = i;
      for(int j = g->former_indegree[from_node]; j < g->indegree[from_node]; j++){
        int to_node = g->inAdjList[from_node][j];

        for(int k = start; k < end; k++){

          if(j == 0){
            continue;
            if(residue_transpose[k][from_node] / j > residuemax && !flags_transpose[k][from_node]){
              enqueue(&queue_list_transpose[k], from_node);
              flags_transpose[k][from_node] = true;
            }

          }


          int col_v = from_node / col_dim;
          
          
          if(col_v == nParts){
            col_v--;
          }

          int inner_col_index = inner_group_mapping[from_node];

          subset_tree->dense_mat_mapping[col_v](
            k, inner_col_index) += pi_transpose[k][from_node] * 1 / j;

          pi_transpose[k][from_node] *= (j + 1) / j;

          residue_transpose[k][from_node] -= pi_transpose[k][from_node] / (j+1) / alpha;
          residue_transpose[k][to_node] += (1 - alpha) * pi_transpose[k][from_node] / (j+1) / alpha;
          if(residue_transpose[k][from_node] / j > residuemax && !flags_transpose[k][from_node]){
            enqueue(&queue_list_transpose[k], from_node);
            flags_transpose[k][from_node] = true;
          }
          if(residue_transpose[k][to_node] / j > residuemax && !flags_transpose[k][to_node]){
            enqueue(&queue_list_transpose[k], to_node);
            flags_transpose[k][to_node] = true;
          }
        }
        
      }
    }
  }


}



















void DenseDirected_Refresh_PPR_initialization_MatrixVersion(int start, int end, Graph* g, double residuemax, double reservemin,
double alpha, vector<int>& labeled_node_vec, 

MatrixXf &residue, 
MatrixXf &pi,
bool** flags, 
Queue* queue_list,

int col_dim,
int nParts,
vector<int> &inner_group_mapping,

int iter,
int vertex_number,

d_row_tree_mkl* subset_tree,
int dynamic_ppr_start_iter
){

  if(iter < dynamic_ppr_start_iter){
    for(int i = start; i < end; i++){

      memset(flags[i], false, sizeof(bool) * vertex_number);
      queue_list[i].front = 0;
      queue_list[i].rear = 0;

      int src = labeled_node_vec[i];

      residue(i,src) = 1;
      flags[i][src] = true;
      enqueue(&queue_list[i], src);
    }
  }
  else{

    for(int i = 0; i < vertex_number; i++){
      int from_node = i;
      for(int j = g->former_outdegree[from_node]; j < g->outdegree[from_node]; j++){
        int to_node = g->outAdjList[from_node][j];
        
        for(int k = start; k < end; k++){

          if(j == 0){
            continue;

            if(residue(k, from_node) / j > residuemax && !flags[k][from_node]){
              enqueue(&queue_list[k], from_node);
              flags[k][from_node] = true;
            }

          }


          int col_v = from_node / col_dim;
          
          if(col_v == nParts){
            col_v--;
          }

          int inner_col_index = inner_group_mapping[from_node];
          
          subset_tree->dense_mat_mapping[col_v](
            k, inner_col_index) += pi(k, from_node) * 1 / j;

          pi(k, from_node) *= (j + 1) / j;

          residue(k, from_node) -= pi(k, from_node) / (j+1) / alpha;
          residue(k, to_node) += (1 - alpha) * pi(k, from_node) / (j+1) / alpha;
          if(residue(k, from_node) / j > residuemax && !flags[k][from_node]){
            enqueue(&queue_list[k], from_node);
            flags[k][from_node] = true;
          }
          if(residue(k, to_node) / j > residuemax && !flags[k][to_node]){
            enqueue(&queue_list[k], to_node);
            flags[k][to_node] = true;
          }
        }

      }
    }
  }




}



















void nodegree_DenseDirected_Refresh_PPR_initialization_MatrixVersion(int start, int end, Graph* g, double residuemax, double reservemin,
double alpha, vector<int>& labeled_node_vec, 

MatrixXf &residue, 
MatrixXf &pi,
bool** flags, 
Queue* queue_list,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,

int iter,
int vertex_number,

d_row_tree_mkl* subset_tree,
int dynamic_ppr_start_iter
){

  if(iter < dynamic_ppr_start_iter){
    for(int i = start; i < end; i++){

      memset(flags[i], false, sizeof(bool) * vertex_number);
      queue_list[i].front = 0;
      queue_list[i].rear = 0;

      int src = labeled_node_vec[i];

      residue(i,src) = 1;
      flags[i][src] = true;
      enqueue(&queue_list[i], src);
    }
  }
  else{

    for(int i = 0; i < vertex_number; i++){
      int from_node = i;
      for(int j = g->former_outdegree[from_node]; j < g->outdegree[from_node]; j++){
        int to_node = g->outAdjList[from_node][j];
        
        for(int k = start; k < end; k++){

          if(j == 0){
            continue;

            if(residue(k, from_node) > residuemax && !flags[k][from_node]){
              enqueue(&queue_list[k], from_node);
              flags[k][from_node] = true;
            }

          }


          int col_v = from_node / col_dim;
          
          if(col_v == nParts){
            col_v--;
          }

          int inner_col_index = inner_group_mapping[from_node];


          subset_tree->dense_mat_mapping[col_v](
            k, inner_col_index) += pi(k, from_node) * 1 / j;

          pi(k, from_node) *= (j + 1) / j;

          residue(k, from_node) -= pi(k, from_node) / (j+1) / alpha;
          residue(k, to_node) += (1 - alpha) * pi(k, from_node) / (j+1) / alpha;

          if(residue(k, from_node) > residuemax && !flags[k][from_node]){
            enqueue(&queue_list[k], from_node);
            flags[k][from_node] = true;
          }

          if(residue(k, to_node) > residuemax && !flags[k][to_node]){
            enqueue(&queue_list[k], to_node);
            flags[k][to_node] = true;
          }
        }

      }
    }
  }




}



























void DenseDirected_Refresh_PPR_initialization_Transpose_MatrixVersion(int start, int end, Graph* g, double residuemax, double reservemin,
double alpha, vector<int>& labeled_node_vec, 

MatrixXf &residue_transpose, 
MatrixXf &pi_transpose,
bool** flags_transpose, 
Queue * queue_list_transpose,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,

int iter,
int vertex_number,

d_row_tree_mkl* subset_tree,
int dynamic_ppr_start_iter
){

  if(iter < dynamic_ppr_start_iter){
    for(int i = start; i < end; i++){

      memset(flags_transpose[i], false, sizeof(bool) * vertex_number);

      queue_list_transpose[i].front = 0;
      queue_list_transpose[i].rear = 0;

      int src = labeled_node_vec[i];

      residue_transpose(i, src) = 1;
      flags_transpose[i][src] = true;
      enqueue(&queue_list_transpose[i], src);
    }

  }
  else{
    for(int i = 0; i < vertex_number; i++){
      int from_node = i;
      for(int j = g->former_indegree[from_node]; j < g->indegree[from_node]; j++){
        int to_node = g->inAdjList[from_node][j];

        for(int k = start; k < end; k++){
          if(j == 0){
            continue;

            if(residue_transpose(k, from_node) / j > residuemax && !flags_transpose[k][from_node]){
              enqueue(&queue_list_transpose[k], from_node);
              flags_transpose[k][from_node] = true;
            }

          }


          int col_v = from_node / col_dim;
          
          if(col_v == nParts){
            col_v--;
          }

          int inner_col_index = inner_group_mapping[from_node];


          subset_tree->dense_mat_mapping[col_v](
            k, inner_col_index) += pi_transpose(k, from_node) * 1 / j;

          pi_transpose(k, from_node) *= (j + 1) / j;

          residue_transpose(k, from_node) -= pi_transpose(k, from_node) / (j+1) / alpha;
          residue_transpose(k, to_node) += (1 - alpha) * pi_transpose(k, from_node) / (j+1) / alpha;
          if(residue_transpose(k, from_node) / j > residuemax && !flags_transpose[k][from_node]){
            enqueue(&queue_list_transpose[k], from_node);
            flags_transpose[k][from_node] = true;
          }
          if(residue_transpose(k, to_node) / j > residuemax && !flags_transpose[k][to_node]){
            enqueue(&queue_list_transpose[k], to_node);
            flags_transpose[k][to_node] = true;
          }
        }
        
      }
    }
  }


}





































void nodegree_DenseDirected_Refresh_PPR_initialization_Transpose_MatrixVersion(int start, int end, Graph* g, double residuemax, double reservemin,
double alpha, vector<int>& labeled_node_vec, 

MatrixXf &residue_transpose, 
MatrixXf &pi_transpose,
bool** flags_transpose, 
Queue * queue_list_transpose,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,

int iter,
int vertex_number,

d_row_tree_mkl* subset_tree,
int dynamic_ppr_start_iter
){

  if(iter < dynamic_ppr_start_iter){
    for(int i = start; i < end; i++){

      memset(flags_transpose[i], false, sizeof(bool) * vertex_number);

      queue_list_transpose[i].front = 0;
      queue_list_transpose[i].rear = 0;

      int src = labeled_node_vec[i];

      residue_transpose(i, src) = 1;
      flags_transpose[i][src] = true;
      enqueue(&queue_list_transpose[i], src);
    }

  }
  else{
    for(int i = 0; i < vertex_number; i++){
      int from_node = i;
      for(int j = g->former_indegree[from_node]; j < g->indegree[from_node]; j++){
        int to_node = g->inAdjList[from_node][j];

        for(int k = start; k < end; k++){
          if(j == 0){
            continue;


            if(residue_transpose(k, from_node) > residuemax && !flags_transpose[k][from_node]){
              enqueue(&queue_list_transpose[k], from_node);
              flags_transpose[k][from_node] = true;
            }

          }


          int col_v = from_node / col_dim;
          
          if(col_v == nParts){
            col_v--;
          }

          int inner_col_index = inner_group_mapping[from_node];


          subset_tree->dense_mat_mapping[col_v](
            k, inner_col_index) += pi_transpose(k, from_node) * 1 / j;

          pi_transpose(k, from_node) *= (j + 1) / j;

          residue_transpose(k, from_node) -= pi_transpose(k, from_node) / (j+1) / alpha;
          residue_transpose(k, to_node) += (1 - alpha) * pi_transpose(k, from_node) / (j+1) / alpha;

          if(residue_transpose(k, from_node) > residuemax && !flags_transpose[k][from_node]){
            enqueue(&queue_list_transpose[k], from_node);
            flags_transpose[k][from_node] = true;
          }

          if(residue_transpose(k, to_node) > residuemax && !flags_transpose[k][to_node]){
            enqueue(&queue_list_transpose[k], to_node);
            flags_transpose[k][to_node] = true;
          }
        }
        
      }
    }
  }


}









































void DenseDirectedDynamicForwardPush_MatrixVersion(int start, int end, Graph* g, double residuemax, double reservemin, 
double alpha, vector<int>& labeled_node_vec, 
MatrixXf & residue, 
MatrixXf & pi,
bool** flags, 
Queue* queue_list,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,

d_row_tree_mkl* subset_tree
)
{

  int vertices_number = g->n;  

  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);

      if(g->outdegree[v] == 0){
        flags[it][v] = false;
        queue_list[it].front++;

        continue;
      }

      if(residue(it, v) / g->outdegree[v] > residuemax){
        for(int j = 0; j < g->outdegree[v]; j++){
          int u = g->outAdjList[v][j];

          residue(it, u) += (1-alpha) * residue(it, v) / g->outdegree[v];

          if(g->outdegree[u] == 0){

            continue;
          }
          
          if(residue(it, u) / g->outdegree[u] > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }
        }

        pi(it, v) += alpha * residue(it, v);


        int col_v = v / col_dim;
        
        if(col_v == nParts){
          col_v--;
        }


        int inner_col_index = inner_group_mapping[v];


        subset_tree->dense_mat_mapping[col_v](
          it, inner_col_index) += alpha * residue(it, v);

        residue(it, v) = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }


    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }


  return;

}























void nodegree_DenseDirectedDynamicForwardPush_MatrixVersion(int start, int end, Graph* g, double residuemax, double reservemin, 
double alpha, vector<int>& labeled_node_vec, 
MatrixXf & residue, 
MatrixXf & pi,
bool** flags, 
Queue* queue_list,


int col_dim,

int nParts,
vector<int> &inner_group_mapping,

d_row_tree_mkl* subset_tree
)
{

  int vertices_number = g->n;  

  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);

      if(g->outdegree[v] == 0){
        flags[it][v] = false;
        queue_list[it].front++;

        continue;
      }


      if(residue(it, v) > residuemax){
        for(int j = 0; j < g->outdegree[v]; j++){
          int u = g->outAdjList[v][j];

          residue(it, u) += (1-alpha) * residue(it, v) / g->outdegree[v];

          if(g->outdegree[u] == 0){

            continue;
          }
          

          if(residue(it, u) > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }
        }

        pi(it, v) += alpha * residue(it, v);


        int col_v = v / col_dim;
        
        if(col_v == nParts){
          col_v--;
        }


        int inner_col_index = inner_group_mapping[v];


        subset_tree->dense_mat_mapping[col_v](
          it, inner_col_index) += alpha * residue(it, v);

        residue(it, v) = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }


    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }


  return;

}


































void DenseDirectedDynamicForwardPushTranspose_MatrixVersion(int start, int end, Graph* g, double residuemax, double reservemin, 

double alpha, vector<int>& labeled_node_vec, 

MatrixXf &residue, 
MatrixXf &pi,
bool** flags, 
Queue* queue_list,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,
d_row_tree_mkl* subset_tree
)
{

  int vertices_number = g->n;  

  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);

      if(g->indegree[v] == 0){
        flags[it][v] = false;
        queue_list[it].front++;

        continue;
      }

      if(residue(it, v) / g->indegree[v] > residuemax){
        for(int j = 0; j < g->indegree[v]; j++){
          int u = g->inAdjList[v][j];
          residue(it, u) += (1-alpha) * residue(it, v) / g->indegree[v];

          if(g->indegree[u] == 0){
            continue;
          }
          
          if(residue(it, u) / g->indegree[u] > residuemax && !flags[it][u]){

            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }
        }
        pi(it, v) += alpha * residue(it, v);


        int col_v = v / col_dim;
        
        if(col_v == nParts){
          col_v--;
        }

        int inner_col_index = inner_group_mapping[v];


        subset_tree->dense_mat_mapping[col_v](
          it, inner_col_index) += alpha * residue(it, v);


        residue(it, v) = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }

    queue_list[it].front = 0;
    queue_list[it].rear = 0;

  }


  return;

}






























void nodegree_DenseDirectedDynamicForwardPushTranspose_MatrixVersion(int start, int end, Graph* g, double residuemax, double reservemin, 

double alpha, vector<int>& labeled_node_vec, 

MatrixXf &residue, 
MatrixXf &pi,
bool** flags, 
Queue* queue_list,


int col_dim,

int nParts,
vector<int> &inner_group_mapping,
d_row_tree_mkl* subset_tree
)
{

  int vertices_number = g->n;  

  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);

      if(g->indegree[v] == 0){
        flags[it][v] = false;
        queue_list[it].front++;

        continue;
      }


      if(residue(it, v) > residuemax){
        for(int j = 0; j < g->indegree[v]; j++){
          int u = g->inAdjList[v][j];
          residue(it, u) += (1-alpha) * residue(it, v) / g->indegree[v];

          if(g->indegree[u] == 0){
            continue;
          }
          

          if(residue(it, u) > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }
        }
        pi(it, v) += alpha * residue(it, v);


        int col_v = v / col_dim;
        
        if(col_v == nParts){
          col_v--;
        }

        int inner_col_index = inner_group_mapping[v];


        subset_tree->dense_mat_mapping[col_v](
          it, inner_col_index) += alpha * residue(it, v);


        residue(it, v) = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }

    queue_list[it].front = 0;
    queue_list[it].rear = 0;

  }


  return;

}

















































































































































































































































































































































































































































void DenseDirectedDynamicBackwardPushTranspose_MatrixVersion(int start, int end, Graph* g, double residuemax, double reservemin, 
double alpha, vector<int>& labeled_node_vec, 

MatrixXf &residue, 
MatrixXf &pi,
bool** flags, 
Queue* queue_list,

int row_dim,
int col_dim,
int number_of_d_row_tree,
int nParts,
vector<int> &inner_group_mapping,
vector<d_row_tree_mkl*> &d_row_tree_vec
)
{

  int vertices_number = g->n;  


  for(int it = start; it < end; it++){


    int src = labeled_node_vec[it];



    while(!isEmpty(&queue_list[it])){
      int v = get_front(&queue_list[it]);
      

      if(g->indegree[v] == 0){
        flags[it][v] = false;
        queue_list[it].front++;

        continue;
      }


      if(residue(it, v) > residuemax){
        for(int j = 0; j < g->outdegree[v]; j++){
          int u = g->outAdjList[v][j];
          residue(it, u) += (1-alpha) * residue(it, v) / g->indegree[u];

          if(g->indegree[u] == 0){
            continue;
          }
          
          if(residue(it, u) > residuemax && !flags[it][u]){

            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }
        }
        pi(it, v) += alpha * residue(it, v);
        
        int row_v = it / row_dim;
        int col_v = v / col_dim;
        if(row_v == number_of_d_row_tree){
          row_v--;
        }
        if(col_v == nParts){
          col_v--;
        }

        int inner_col_index = inner_group_mapping[v];

        d_row_tree_vec[row_v]->dense_mat_mapping[col_v](
          it - row_v * row_dim, inner_col_index) += alpha * residue(it, v);

        residue(it, v) = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }



    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }

  return;

}

