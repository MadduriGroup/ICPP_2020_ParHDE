// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <iostream>
#include <vector>
#include "bitmap.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "sliding_queue.h"


/*
GAP Benchmark Suite
Kernel: Breadth-First Search (BFS)
Author: Scott Beamer

Will return parent array for a BFS traversal from a source vertex

This BFS implementation makes use of the Direction-Optimizing approach [1].
It uses the alpha and beta parameters to determine whether to switch search
directions. For representing the frontier, it uses a SlidingQueue for the
top-down approach and a Bitmap for the bottom-up approach. To reduce
false-sharing for the top-down approach, thread-local QueueBuffer's are used.

To save time computing the number of edges exiting the frontier, this
implementation precomputes the degrees in bulk at the beginning by storing
them in parent array as negative numbers. Thus the encoding of parent is:
  parent[x] < 0 implies x is unvisited and parent[x] = -out_degree(x)
  parent[x] >= 0 implies x been visited

[1] Scott Beamer, Krste Asanović, and David Patterson. "Direction-Optimizing
    Breadth-First Search." International Conference on High Performance
    Computing, Networking, Storage and Analysis (SC), Salt Lake City, Utah,
    November 2012.
*/


using namespace std;

typedef long NodeID;

unsigned int get_degree(unsigned int *row, unsigned int s)
{
	return (row[s+1]-row[s]);
}

long BUStep(unsigned int *row, unsigned int *col, long numNodes, pvector<NodeID> &parent, Bitmap &front,
               Bitmap &next, double *dists) 
{
  long awake_count = 0;
  next.reset();
  #pragma omp parallel for reduction(+ : awake_count) schedule(dynamic, 1024)
  for (NodeID u=0; u < numNodes; u++) 
  {
    if (parent[u] < 0) 
    {
      for(unsigned int j=row[u]; j<row[u+1]; j++)
      {
        NodeID v = (long)col[j];
        if (front.get_bit(v)) 
        {
          parent[u] = v;
          dists[u] = dists[v] + 1;
          awake_count++;
          next.set_bit(u);
          break;
        }
      }
    }
  }
  return awake_count;
}


long TDStep(unsigned int *row, unsigned int *col, pvector<NodeID> &parent,
               SlidingQueue<NodeID> &queue, double *dists)
{
  long scout_count = 0;
  #pragma omp parallel
  {
    QueueBuffer<NodeID> lqueue(queue);
    double incrDistance = dists[*(queue.begin())] + 1;
    #pragma omp for reduction(+ : scout_count)
    for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) 
    {
      NodeID u = *q_iter;
      for(unsigned int j=row[u]; j<row[u+1]; j++)
      {
        NodeID v = (long)col[j];
        NodeID curr_val = parent[v];
        if (curr_val < 0) 
        {
          if (compare_and_swap(parent[v], curr_val, u)) 
          {
            lqueue.push_back(v);
            scout_count += -curr_val;
            dists[v] = incrDistance;
          }
        }
      }
    }
    lqueue.flush();
  }
  return scout_count;
}


void QueueToBitmap(const SlidingQueue<NodeID> &queue, Bitmap &bm) 
{
  #pragma omp parallel for
  for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) 
  {
    NodeID u = *q_iter;
    bm.set_bit_atomic(u);
  }
}

void BitmapToQueue(long numNodes, const Bitmap &bm,
                   SlidingQueue<NodeID> &queue) 
{
  #pragma omp parallel
  {
    QueueBuffer<NodeID> lqueue(queue);
    #pragma omp for
    for (NodeID n=0; n < numNodes; n++)
      if (bm.get_bit(n))
        lqueue.push_back(n);
    lqueue.flush();
  }
  queue.slide_window();
}

pvector<NodeID> InitParent(long numNodes, unsigned int *row) 
{
  pvector<NodeID> parent(numNodes);
  #pragma omp parallel for
  for (NodeID n=0; n < numNodes; n++)
    parent[n] = get_degree(row,n) != 0 ? -(long)get_degree(row,n) : -1;
  return parent;
}


void DOBFS(unsigned int *row, unsigned int *col, NodeID source, long numNodes, long numEdges, double *dists,
		      int alpha = 15, int beta = 18) 
{
  pvector<NodeID> parent = InitParent(numNodes, row);
  parent[source] = source;
  SlidingQueue<NodeID> queue(numNodes);
  queue.push_back(source);
  dists[source] = 0;
  queue.slide_window();
  Bitmap curr(numNodes);
  curr.reset();
  Bitmap front(numNodes);
  front.reset();
  long edges_to_check = numEdges;
  long scout_count = get_degree(row,source);
  while (!queue.empty()) 
  {
    if (scout_count > edges_to_check / alpha) 
    {
      long awake_count, old_awake_count;
      QueueToBitmap(queue, front);
      awake_count = (long)queue.size();
      queue.slide_window();
      do 
      {
        old_awake_count = awake_count;
        awake_count = BUStep(row, col, numNodes, parent, front, curr, dists);
        front.swap(curr);
      } while ((awake_count >= old_awake_count) ||
               (awake_count > numNodes /( (long)beta)));
      BitmapToQueue(numNodes, front, queue);
      scout_count = 1;
    } 
    else 
    {
      edges_to_check -= scout_count;
      scout_count = TDStep(row, col, parent, queue, dists);
      queue.slide_window();
    }
  }
  return;
}