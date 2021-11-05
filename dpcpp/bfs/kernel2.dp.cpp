#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "bfs.hpp"

/*********************************************************************************
Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

Copyright (c) 2008 International Institute of Information Technology -
Hyderabad. All rights reserved.

Permission to use, copy, modify and distribute this software and its
documentation for educational purpose is hereby granted without fee, provided
that the above copyright notice and this permission notice appear in all copies
of this software and that you do not sell the software.

THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS,
IMPLIED OR OTHERWISE.

The CUDA Kernel for Applying BFS on a loaded Graph. Created By Pawan Harish
**********************************************************************************/
#ifndef _KERNEL2_H_
#define _KERNEL2_H_

void
Kernel2( bool* g_graph_mask, bool *g_updating_graph_mask, bool* g_graph_visited, bool *g_over, int no_of_nodes,
         sycl::nd_item<3> item_ct1, int max_blocks)
{
		int MAX_THREADS_PER_BLOCK = max_blocks;
        int tid = item_ct1.get_group(2) * MAX_THREADS_PER_BLOCK +
                  item_ct1.get_local_id(2);
        if( tid<no_of_nodes && g_updating_graph_mask[tid])
	{

		g_graph_mask[tid]=true;
		g_graph_visited[tid]=true;
		*g_over=true;
		g_updating_graph_mask[tid]=false;
	}
}

#endif

