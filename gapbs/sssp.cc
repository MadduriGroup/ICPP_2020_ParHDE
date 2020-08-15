// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <cinttypes>
#include <limits>
#include <iostream>
#include <queue>
#include <vector>
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"


/*
GAP Benchmark Suite
Kernel: Single-source Shortest Paths (SSSP)
Author: Scott Beamer

Returns array of distances for all vertices from given source vertex

This SSSP implementation makes use of the ∆-stepping algorithm [1]. The type
used for weights and distances (WeightT) is typedefined in benchmark.h. The
delta parameter (-d) should be set for each input graph.

The bins of width delta are actually all thread-local and of type std::vector
so they can grow but are otherwise capacity-proportional. Each iteration is
done in two phases separated by barriers. In the first phase, the current
shared bin is processed by all threads. As they find vertices whose distance
they are able to improve, they add them to their thread-local bins. During this
phase, each thread also votes on what the next bin should be (smallest
non-empty bin). In the next phase, each thread copies their selected
thread-local bin into the shared bin.

Once a vertex is added to a bin, it is not removed, even if its distance is
later updated and it now appears in a lower bin. We find ignoring vertices if
their current distance is less than the min distance for the bin to remove
enough redundant work that this is faster than removing the vertex from older
bins.

[1] Ulrich Meyer and Peter Sanders. "δ-stepping: a parallelizable shortest path
    algorithm." Journal of Algorithms, 49(1):114–152, 2003.
*/


using namespace std;

typedef long NodeID;
typedef int32_t WeightT;

const WeightT kDistInf = numeric_limits<WeightT>::max()/2;
const size_t kMaxBin = numeric_limits<size_t>::max()/2;

double *  DeltaStep(const unsigned int *row, unsigned int *col, int32_t *weights,
		    NodeID source, long numNodes, long numEdges, double *dists,
 		    WeightT delta) 
{
  pvector<WeightT> dist(numNodes, kDistInf);
  dist[source] = 0;
  pvector<NodeID> frontier(numEdges);
  // two element arrays for double buffering curr=iter&1, next=(iter+1)&1
  size_t shared_indexes[2] = {0, kMaxBin};
  size_t frontier_tails[2] = {1, 0};
  frontier[0] = source;
  #pragma omp parallel
  {
    vector<vector<NodeID> > local_bins(0);
    size_t iter = 0;
    while (shared_indexes[iter&1] != kMaxBin) 
    {
      size_t &curr_bin_index = shared_indexes[iter&1];
      size_t &next_bin_index = shared_indexes[(iter+1)&1];
      size_t &curr_frontier_tail = frontier_tails[iter&1];
      size_t &next_frontier_tail = frontier_tails[(iter+1)&1];
      #pragma omp for nowait schedule(dynamic, 64)
      for (size_t i=0; i < curr_frontier_tail; i++) 
      {
        NodeID u = frontier[i];
        if (dist[u] >= delta * static_cast<WeightT>(curr_bin_index)) 
        {
          for(unsigned int j = row[u]; j < row[u+1]; j++)
          {
            WeightT old_dist = dist[col[j]];
            WeightT new_dist = dist[u] + weights[j];
            if (new_dist < old_dist) 
            {
              bool changed_dist = true;
              while (!compare_and_swap(dist[col[j]], old_dist, new_dist)) 
              {
                old_dist = dist[col[j]];
                if (old_dist <= new_dist) 
                {
                  changed_dist = false;
                  break;
                }
              }
              if (changed_dist) 
              {
                size_t dest_bin = new_dist/delta;
                if (dest_bin >= local_bins.size()) 
                {
                  local_bins.resize(dest_bin+1);
                }
                local_bins[dest_bin].push_back(col[j]);
              }
            }
          }
        }
      }
      for (size_t i=curr_bin_index; i < local_bins.size(); i++) 
      {
        if (!local_bins[i].empty()) 
        {
          #pragma omp critical
          next_bin_index = min(next_bin_index, i);
          break;
        }
      }
      #pragma omp barrier
      #pragma omp single nowait
      {
        curr_bin_index = kMaxBin;
        curr_frontier_tail = 0;
      }
      if (next_bin_index < local_bins.size()) 
      {
        size_t copy_start = fetch_and_add(next_frontier_tail,
                                          local_bins[next_bin_index].size());
        copy(local_bins[next_bin_index].begin(),
             local_bins[next_bin_index].end(), frontier.data() + copy_start);
        local_bins[next_bin_index].resize(0);
      }
      iter++;
      #pragma omp barrier
    }
  }
  #pragma omp parallel for 
  for(long p=0; p < numNodes; p++)
     dists[p] = dist[p];
  return dists;
}
