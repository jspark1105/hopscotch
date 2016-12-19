#include <cstdio>
#include <cfloat>
#include <cassert>
#include <algorithm>

#include <vector>
#include <unordered_set>
#include <unordered_map>

#include <omp.h>

#include <tbb/concurrent_unordered_set.h>
#include <tbb/concurrent_unordered_map.h>

#include "hopscotch_hash.h"

#define VERBOSE
#define CNT // count # of operations

#define SWAP(T, a, b) do { T tmp = a; a = b; b = tmp; } while (0)

static void merge(int *first1, int *last1, int *first2, int *last2, int *out)
{
   for ( ; first1 != last1; ++out)
   {
      if (first2 == last2)
      {
         for ( ; first1 != last1; ++first1, ++out)
         {
            *out = *first1;
         }
         return;
      }
      if (*first2 < *first1)
      {
         *out = *first2;
         ++first2;
      }
      else
      {
         *out = *first1;
         ++first1;
      }
   }
   for ( ; first2 != last2; ++first2, ++out)
   {
      *out = *first2;
   }
}

/**
 * merge two sorted (w/o duplication) sequences and eliminate
 * duplicates.
 *
 * @return length of merged output
 */
int *merge_unique(int *first1, int *last1, int *first2, int *last2, int *out)
{
   for ( ; first1 != last1; ++out)
   {
      if (first2 == last2)
      {
         for ( ; first1 != last1; ++out)
         {
            *out = *first1;
            ++first1;
            for ( ; first1 != last1 && *first1 == *out; ++first1);
         }
         return out;
      }
      if (*first1 == *first2)
      {
         *out = *first1;
         ++first1; ++first2;
         for ( ; first1 != last1 && *first1 == *out; ++first1);
         for ( ; first2 != last2 && *first2 == *out; ++first2);
      }
      else if (*first2 < *first1)
      {
         *out = *first2;
         ++first2;
         for ( ; first2 != last2 && *first2 == *out; ++first2);
      }
      else
      {
         *out = *first1;
         ++first1;
         for ( ; first1 != last1 && *first1 == *out; ++first1);
      }
   }
   for ( ; first2 != last2; ++out)
   {
      *out = *first2;
      ++first2;
      for ( ; first2 != last2 && *first2 == *out; ++first2);
   }
   return out;
}


static void kth_element_(
   int *out1, int *out2,
   int *a1, int *a2,
   int left, int right, int n1, int n2, int k)
{
   while (1)
   {
      int i = (left + right)/2; // right < k -> i < k
      int j = k - i - 1;
#ifdef DBG_MERGE_SORT
      assert(left <= right && right <= k);
      assert(i < k); // i == k implies left == right == k that can never happen
      assert(j >= 0 && j < n2);
#endif

      if ((j == -1 || a1[i] >= a2[j]) && (j == n2 - 1 || a1[i] <= a2[j + 1]))
      {
         *out1 = i; *out2 = j + 1;
         return;
      }
      else if (j >= 0 && a2[j] >= a1[i] && (i == n1 - 1 || a2[j] <= a1[i + 1]))
      {
         *out1 = i + 1; *out2 = j;
         return;
      }
      else if (a1[i] > a2[j] && j != n2 - 1 && a1[i] > a2[j+1])
      {
         // search in left half of a1
         right = i - 1;
      }
      else
      {
         // search in right half of a1
         left = i + 1;
      }
   }
}

/**
 * Partition the input so that
 * a1[0:*out1) and a2[0:*out2) contain the smallest k elements
 */
static void kth_element(
   int *out1, int *out2,
   int *a1, int *a2, int n1, int n2, int k)
{
   // either of the inputs is empty
   if (n1 == 0)
   {
      *out1 = 0; *out2 = k;
      return;
   }
   if (n2 == 0)
   {
      *out1 = k; *out2 = 0;
      return;
   }
   if (k >= n1 + n2)
   {
      *out1 = n1; *out2 = n2;
      return;
   }

   // one is greater than the other
   if (k < n1 && a1[k] <= a2[0])
   {
      *out1 = k; *out2 = 0;
      return;
   }
   if (k - n1 >= 0 && a2[k - n1] >= a1[n1 - 1])
   {
      *out1 = n1; *out2 = k - n1;
      return;
   }
   if (k < n2 && a2[k] <= a1[0])
   {
      *out1 = 0; *out2 = k;
      return;
   }
   if (k - n2 >= 0 && a1[k - n2] >= a2[n2 - 1])
   {
      *out1 = k - n2; *out2 = n2;
      return;
   }
   // now k > 0

   // faster to do binary search on the shorter sequence
   if (n1 > n2)
   {
      SWAP(int, n1, n2);
      SWAP(int *, a1, a2);
      SWAP(int *, out1, out2);
   }

   if (k < (n1 + n2)/2)
   {
      kth_element_(out1, out2, a1, a2, 0, std::min(n1 - 1, k), n1, n2, k);
   }
   else
   {
      // when k is big, faster to find (n1 + n2 - k)th biggest element
      int offset1 = std::max(k - n2, 0), offset2 = std::max(k - n1, 0);
      int new_k = k - offset1 - offset2;

      int new_n1 = std::min(n1 - offset1, new_k + 1);
      int new_n2 = std::min(n2 - offset2, new_k + 1);
      kth_element_(out1, out2, a1 + offset1, a2 + offset2, 0, new_n1 - 1, new_n1, new_n2, new_k);

      *out1 += offset1;
      *out2 += offset2;
   }
#ifdef DBG_MERGE_SORT
   assert(*out1 + *out2 == k);
#endif
}

/**
 * @param num_threads number of threads that participate in this merge
 * @param my_thread_num thread id (zeor-based) among the threads that participate in this merge
 */
static void parallel_merge(
   int *first1, int *last1, int *first2, int *last2,
   int *out,
   int num_threads, int my_thread_num)
{
   int n1 = last1 - first1;
   int n2 = last2 - first2;
   int n = n1 + n2;
   int n_per_thread = (n + num_threads - 1)/num_threads;
   int begin_rank = std::min(n_per_thread*my_thread_num, n);
   int end_rank = std::min(begin_rank + n_per_thread, n);

#ifdef DBG_MERGE_SORT
   assert(std::is_sorted(first1, last1));
   assert(std::is_sorted(first2, last2));
#endif

   int begin1, begin2, end1, end2;
   kth_element(&begin1, &begin2, first1, first2, n1, n2, begin_rank);
   kth_element(&end1, &end2, first1, first2, n1, n2, end_rank);

   while (begin1 > end1 && begin1 > 0 && begin2 < n2 && first1[begin1 - 1] == first2[begin2])
   {
#ifdef DBG_MERGE_SORT
      printf("%s:%d\n", __FILE__, __LINE__);
#endif
      begin1--; begin2++; 
   }
   while (begin2 > end2 && end1 > 0 && end2 < n2 && first1[end1 - 1] == first2[end2])
   {
#ifdef DBG_MERGE_SORT
      printf("%s:%d\n", __FILE__, __LINE__);
#endif
      end1--; end2++;
   }

#ifdef DBG_MERGE_SORT
   assert(begin1 <= end1);
   assert(begin2 <= end2);
#endif

   merge(
      first1 + begin1, first1 + end1,
      first2 + begin2, first2 + end2,
      out + begin1 + begin2);

#ifdef DBG_MERGE_SORT
   assert(std::is_sorted(out + begin1 + begin2, out + end1 + end2));
#endif
}

/**
 * @param num_threads number of threads that participate in this merge
 * @param my_thread_num thread id (zeor-based) among the threads that participate in this merge
 *
 * It is assumed that [first1, last1) and [first2, last2) are sorted
 * without dulication.
 *
 * @return length of merged output
 */
int parallel_merge_unique(
   int *first1, int *last1, int *first2, int *last2,
   int *temp, int *out,
   int num_threads, int my_thread_num,
   int *prefix_sum_workspace)
{
   int n1 = last1 - first1;
   int n2 = last2 - first2;
   int n = n1 + n2;
   int n_per_thread = (n + num_threads - 1)/num_threads;
   int begin_rank = std::min(n_per_thread*my_thread_num, n);
   int end_rank = std::min(begin_rank + n_per_thread, n);

#ifdef DBG_MERGE_SORT
   int *dbg_buf = NULL;
   if (my_thread_num == 0)
   {
      dbg_buf = new int[n];
      std::merge(first1, last1, first2, last2, dbg_buf);
   }
#endif

   int begin1, begin2, end1, end2;
   kth_element(&begin1, &begin2, first1, first2, n1, n2, begin_rank);
   kth_element(&end1, &end2, first1, first2, n1, n2, end_rank);

   while (begin1 > end1 && begin1 > 0 && begin2 < n2 && first1[begin1 - 1] == first2[begin2])
   {
#ifdef DBG_MERGE_SORT
      printf("%s:%d\n", __FILE__, __LINE__);
#endif
      begin1--; begin2++; 
   }
   while (begin2 > end2 && end1 > 0 && end2 < n2 && first1[end1 - 1] == first2[end2])
   {
#ifdef DBG_MERGE_SORT
      printf("%s:%d\n", __FILE__, __LINE__);
#endif
      end1--; end2++;
   }

#ifdef DBG_MERGE_SORT
   assert(begin1 <= end1);
   assert(begin2 <= end2);
#endif

   int out_len = merge_unique(
      first1 + begin1, first1 + end1,
      first2 + begin2, first2 + end2,
      temp + begin1 + begin2) - (temp + begin1 + begin2);

#ifdef DBG_MERGE_SORT
   assert(std::is_sorted(temp + begin1 + begin2, temp + begin1 + begin2 + out_len));
   assert(std::adjacent_find(temp + begin1 + begin2, temp + begin1 + begin2 + out_len) == temp + begin1 + begin2 + out_len);
#endif

   prefix_sum_workspace[my_thread_num] = out_len;

#ifdef _OPENMP
#pragma omp barrier
#endif

   int i;
   if (0 == my_thread_num)
   {
      // take care of duplicates at boundary
      int prev = temp[out_len - 1];
      int t;
      for (t = 1; t < num_threads; t++)
      {
         int begin_rank = std::min(n_per_thread*t, n);
         int end_rank = begin_rank + prefix_sum_workspace[t];
         for (i = begin_rank; i < end_rank && temp[i] == prev; i++);
         prefix_sum_workspace[t] -= i - begin_rank;
         prefix_sum_workspace[t] += prefix_sum_workspace[t - 1];

         if (prefix_sum_workspace[t] > 0)
         {
            prev = temp[end_rank - 1];
         }
      }
   }

#ifdef _OPENMP
#pragma omp barrier
#endif

   int out_begin = my_thread_num == 0 ? 0 : prefix_sum_workspace[my_thread_num - 1];
   int out_end = prefix_sum_workspace[my_thread_num];

   int num_duplicates = out_len - (out_end - out_begin);
   begin_rank += num_duplicates;

   for (i = 0; i < out_end - out_begin; i++)
   {
      out[i + out_begin] = temp[i + begin_rank];
   }

#ifdef DBG_MERGE_SORT
   assert(std::is_sorted(out + out_begin, out + out_end));
   assert(std::adjacent_find(out + out_begin, out + out_end) == out + out_end);

#ifdef _OPENMP
#pragma omp barrier
#endif

   if (0 == my_thread_num)
   {
      assert(std::is_sorted(out, out + prefix_sum_workspace[num_threads - 1]));
      assert(std::adjacent_find(out, out + prefix_sum_workspace[num_threads - 1]) == out + prefix_sum_workspace[num_threads - 1]);
      int ref_len = std::unique(dbg_buf, dbg_buf + n) - dbg_buf;
      assert(ref_len == prefix_sum_workspace[num_threads - 1]);
      assert(std::equal(out, out + prefix_sum_workspace[num_threads - 1], dbg_buf));

      delete[] dbg_buf;
   }

#ifdef _OPENMP
#pragma omp barrier
#endif
#endif

   return prefix_sum_workspace[num_threads - 1];
}


void merge_sort(int *in, int *temp, int len, int **out)
{
   if (0 == len) return;

#ifdef DBG_MERGE_SORT
   int *dbg_buf = new int[len];
   std::copy(in, in + len, dbg_buf);
   std::sort(dbg_buf, dbg_buf + len);
#endif

   int thread_private_len[omp_get_max_threads()];
   int out_len = 0;

#ifdef _OPENMP
#pragma omp parallel
#endif
   {
      int num_threads = omp_get_num_threads();
      int my_thread_num = omp_get_thread_num();

      // thread-private sort
      int i_per_thread = (len + num_threads - 1)/num_threads;
      int i_begin = std::min(i_per_thread*my_thread_num, len);
      int i_end = std::min(i_begin + i_per_thread, len);

      std::sort(in + i_begin, in + i_end);

      // merge sorted sequences
      int in_group_size;
      int *in_buf = in;
      int *out_buf = temp;
      for (in_group_size = 1; in_group_size < num_threads; in_group_size *= 2)
      {
#ifdef _OPENMP
#pragma omp barrier
#endif

         // merge 2 in-groups into 1 out-group
         int out_group_size = in_group_size*2;
         int group_leader = my_thread_num/out_group_size*out_group_size;
         int group_sub_leader = std::min(group_leader + in_group_size, num_threads - 1);
         int id_in_group = my_thread_num%out_group_size;
         int num_threads_in_group =
            std::min(group_leader + out_group_size, num_threads) - group_leader;

         int in_group1_begin = std::min(i_per_thread*group_leader, len);
         int in_group1_end = std::min(in_group1_begin + i_per_thread*in_group_size, len);

         int in_group2_begin = std::min(in_group1_begin + i_per_thread*in_group_size, len);
         int in_group2_end = std::min(in_group2_begin + i_per_thread*in_group_size, len);

         parallel_merge(
            in_buf + in_group1_begin, in_buf + in_group1_end,
            in_buf + in_group2_begin, in_buf + in_group2_end,
            out_buf + in_group1_begin,
            num_threads_in_group,
            id_in_group);

         int *temp = in_buf;
         in_buf = out_buf;
         out_buf = temp;
      }

      *out = in_buf;
   } /* omp parallel */

#ifdef DBG_MERGE_SORT
   assert(std::equal(*out, *out + len, dbg_buf));

   delete[] dbg_buf;
#endif
}

/**
 * @params in contents can change
 */
int merge_sort_unique2(int *in, int *temp, int len, int **out)
{
   if (0 == len) return 0;

#ifdef DBG_MERGE_SORT
   int *dbg_buf = new int[len];
   std::copy(in, in + len, dbg_buf);
   std::sort(dbg_buf, dbg_buf + len);
#endif

   int thread_private_len[omp_get_max_threads()];
   int out_len = 0;

#ifdef _OPENMP
#pragma omp parallel
#endif
   {
      int num_threads = omp_get_num_threads();
      int my_thread_num = omp_get_thread_num();

      // thread-private sort
      int i_per_thread = (len + num_threads - 1)/num_threads;
      int i_begin = std::min(i_per_thread*my_thread_num, len);
      int i_end = std::min(i_begin + i_per_thread, len);

      std::sort(in + i_begin, in + i_end);

      if (1 == num_threads)
      {
         // when there's only one thread, we need to eliminate duplicates separately
         if (len)
         {
            out_len = 1;

            int i;
            for (i = 1; i < len; i++)
            {
               if (in[i] != in[i - 1]) in[out_len++] = in[i];
            }
         }

         *out = in;
      }
      else
      {
         // merge sorted sequences
         int in_group_size;
         int *in_buf = in;
         int *out_buf = temp;
         for (in_group_size = 1; in_group_size < num_threads; in_group_size *= 2)
         {
#ifdef _OPENMP
#pragma omp barrier
#endif

            // merge 2 in-groups into 1 out-group
            int out_group_size = in_group_size*2;
            int group_leader = my_thread_num/out_group_size*out_group_size;
            int group_sub_leader = std::min(group_leader + in_group_size, num_threads - 1);
            int id_in_group = my_thread_num%out_group_size;
            int num_threads_in_group =
               std::min(group_leader + out_group_size, num_threads) - group_leader;

            int in_group1_begin = std::min(i_per_thread*group_leader, len);
            int in_group1_end = std::min(in_group1_begin + i_per_thread*in_group_size, len);

            int in_group2_begin = std::min(in_group1_begin + i_per_thread*in_group_size, len);
            int in_group2_end = std::min(in_group2_begin + i_per_thread*in_group_size, len);

            if (out_group_size < num_threads)
            {
               parallel_merge(
                  in_buf + in_group1_begin, in_buf + in_group1_end,
                  in_buf + in_group2_begin, in_buf + in_group2_end,
                  out_buf + in_group1_begin,
                  num_threads_in_group,
                  id_in_group);
            }
            else
            {
               int len = parallel_merge_unique(
                  in_buf + in_group1_begin, in_buf + in_group1_end,
                  in_buf + in_group2_begin, in_buf + in_group2_end,
                  out_buf + in_group1_begin,
                  in_buf + in_group1_begin,
                  num_threads_in_group,
                  id_in_group,
                  thread_private_len + group_leader);

               if (0 == my_thread_num) out_len = len;
            }

            int *temp = in_buf;
            in_buf = out_buf;
            out_buf = temp;
         }

         *out = out_buf;
      }
   } /* omp parallel */

#ifdef DBG_MERGE_SORT
   int ref_len = std::unique(dbg_buf, dbg_buf + len) - dbg_buf;
   assert(ref_len == out_len);
   assert(std::equal(*out, *out + out_len, dbg_buf));

   delete[] dbg_buf;
#endif

   return out_len;
}


int sort_unique_and_inverse_map(
  int *in, int len, int **out, HopscotchUnorderedIntMap *inverse_map)
{
   if (len == 0)
   {
      return 0;
   }

   int *temp = (int *)malloc(sizeof(int)*len);
   int *duplicate_eliminated;
   int new_len = merge_sort_unique2(in, temp, len, &duplicate_eliminated);
   hopscotchUnorderedIntMapCreate(inverse_map, 2*new_len, 16*omp_get_max_threads());
   int i;
   for (i = 0; i < new_len; i++)
   {
      hopscotchUnorderedIntMapPutIfAbsent(inverse_map, duplicate_eliminated[i], i);
   }
   if (duplicate_eliminated == in)
   {
      free(temp);
   }
   else
   {
      free(in);
   }
   *out = duplicate_eliminated;
   return new_len;
}


void sort_and_create_inverse_map(
  int *in, int len, int **out, HopscotchUnorderedIntMap *inverse_map)
{
   if (len == 0)
   {
      return;
   }

   int *temp = (int *)malloc(sizeof(int)*len);
   merge_sort(in, temp, len, out);
   hopscotchUnorderedIntMapCreate(inverse_map, 2*len, 16*omp_get_max_threads());
   int i;
#ifdef CONCURRENT_HOPSCOTCH
#pragma omp parallel for
#endif
   for (i = 0; i < len; i++)
   {
      int old = hopscotchUnorderedIntMapPutIfAbsent(inverse_map, (*out)[i], i);
      assert(old == HOPSCOTCH_HASH_EMPTY);
#ifdef DBG_MERGE_SORT
      if (hopscotchUnorderedIntMapGet(inverse_map, (*out)[i]) != i)
      {
         fprintf(stderr, "%d %d\n", i, (*out)[i]);
         assert(false);
      }
#endif
   }

#ifdef DBG_MERGE_SORT
  std::unordered_map<int, int> inverse_map2(len);
  for (int i = 0; i < len; ++i) {
    inverse_map2[(*out)[i]] = i;
    if (hopscotchUnorderedIntMapGet(inverse_map, (*out)[i]) != i)
    {
      fprintf(stderr, "%d %d\n", i, (*out)[i]);
      assert(false);
    }
  }
  assert(HopscotchUnorderedIntMapSize(inverse_map) == len);
#endif

   if (*out == in)
   {
      free(temp);
   }
   else
   {
      free(in);
   }
}

using namespace std;

int col_1, col_n, num_cols_A_offd;
int *col_map_offd, *CF_marker_offd, *A_ext_i, *Sop_i, *A_ext_j, *Sop_j;

bool first = true;
int lengthExpected = -1;
int sumExpected = -1;

void hashItBaseline()
{
  double t = omp_get_wtime();
  // Step 1: construct inverse map of col_map_offd_inverse
  unordered_map<int, int> col_map_offd_inverse(num_cols_A_offd);
  for (int i = 0; i < num_cols_A_offd; ++i) {
    col_map_offd_inverse[col_map_offd[i]] = i;
  }

  double inverse_col_map_offd_time = omp_get_wtime() - t; t = omp_get_wtime();

  // Step 2: insert A_ext_js and Sop_js into found_set
  int prefix_sum_workspace[omp_get_max_threads() + 1];
  int newoff;

  unordered_set<int> found_set;
  double create_hash_time = omp_get_wtime() - t; t = omp_get_wtime();

  int map_lookup_cnt = 0, set_lookup_cnt = 0, set_insert_cnt = 0;

  for (int i = 0; i < num_cols_A_offd; ++i) {
    if (CF_marker_offd[i] < 0) {
      for (int j = A_ext_i[i]; j < A_ext_i[i+1]; ++j) {
        int i1 = A_ext_j[j];
        if (i1 < col_1 || i1 >= col_n) {
#ifdef CNT
          ++set_lookup_cnt;
#endif
          if (found_set.find(i1) == found_set.end()) {
            auto itr = col_map_offd_inverse.find(i1);
#ifdef CNT
            ++map_lookup_cnt;
#endif
            if (itr == col_map_offd_inverse.end()) {
              found_set.insert(i1);
#ifdef CNT
              ++set_insert_cnt;
#endif
            }
            else {
              A_ext_j[j] = -itr->second - 1;
            }
          }
        }
      }
      for (int j = Sop_i[i]; j < Sop_i[i+1]; ++j) {
        int i1 = Sop_j[j];
        if (i1 < col_1 || i1 >= col_n) {
#ifdef CNT
          ++set_lookup_cnt;
#endif
          if (found_set.find(i1) == found_set.end()) {
            Sop_j[j] = -col_map_offd_inverse[i1] - 1;
#ifdef CNT
            ++map_lookup_cnt;
#endif
          }
        }
      }
    } // CF_marker_offd[i] < 0
  } // for each row

  double insert_hash_time = omp_get_wtime() - t; t = omp_get_wtime();

  // Step 3: copy found_set to an array found
  int i = 0;
  newoff = found_set.size();
  int *found = new int[found_set.size()];
  for (auto itr = found_set.begin(); itr != found_set.end(); ++itr, ++i) {
    found[i] = *itr;
  }

  double found_time = omp_get_wtime() - t; t = omp_get_wtime();

#ifdef VERBOSE
  printf("*Baseline: ninput: %d noutput: %d\n", A_ext_i[num_cols_A_offd], newoff);
#endif

  // Step 4: sort found
  int *temp = new int[newoff];
  int *sorted;
  merge_sort(found, temp, newoff, &sorted);
  if (sorted == found) {
    delete[] temp;
  }
  else {
    free(found);
  }

  double merge_time = omp_get_wtime() - t; t = omp_get_wtime();

  // Step 5: construct inverse map of found
  unordered_map<int, int> found_inverse(newoff);
  for (int i = 0; i < newoff; ++i) {
    found_inverse[sorted[i]] = i;
  }
  found = sorted;

  double inverse_found_time = omp_get_wtime() - t; t = omp_get_wtime();

  // Step 6: fill Sop_j and A_ext_i with new indices
  int lookup_inverse_cnt = 0;
  for (int i = 0; i < num_cols_A_offd; ++i) {
    if (CF_marker_offd[i] < 0) {
      for (int kk = Sop_i[i]; kk < Sop_i[i + 1]; ++kk) {
        int k1 = Sop_j[kk];
        if (k1 > -1 && (k1 < col_1 || k1 >= col_n)) {
          int got_loc = found_inverse[k1];
          int loc_col = got_loc + num_cols_A_offd;
          Sop_j[kk] = -loc_col - 1;
#ifdef CNT
          ++lookup_inverse_cnt;
#endif
        }
      }
      for (int kk = A_ext_i[i]; kk < A_ext_i[i + 1]; ++kk) {
        int k1 = A_ext_j[kk];
        if (k1 > -1 && (k1 < col_1 || k1 >= col_n)) {
          int got_loc = found_inverse[k1];
          int loc_col = got_loc + num_cols_A_offd;
          A_ext_j[kk] = -loc_col - 1;
#ifdef CNT
          ++lookup_inverse_cnt;
#endif
        }
      }
    }
  }

  double lookup_inverse_found_time = omp_get_wtime() - t;

#ifdef VERBOSE
  printf(
    "inverse_col_map_offd takes %f (throughput %g)\n",
    inverse_col_map_offd_time, num_cols_A_offd/inverse_col_map_offd_time);
  printf(
    "create_hash takes %f (throughput %g)\n",
    create_hash_time, A_ext_i[num_cols_A_offd]/create_hash_time);
  printf(
    "insert_hash takes %f (throughput %g set_lookup %d set_insert %d map_lookup %d)\n",
    insert_hash_time,
    (set_lookup_cnt + set_insert_cnt + map_lookup_cnt)/insert_hash_time,
    set_lookup_cnt,
    set_insert_cnt,
    map_lookup_cnt);
  printf(
    "found takes %f (throughput %g)\n", found_time, A_ext_i[num_cols_A_offd]/found_time);
  printf(
    "merge takes %f (throughput %g)\n", merge_time, newoff/merge_time);
  printf(
    "inverse_found_time takes %f (throughput %g)\n", inverse_found_time, newoff/inverse_found_time);
  printf(
    "lookup_inverse_found_time takes %f (throughput %g)\n\n", lookup_inverse_found_time, lookup_inverse_cnt/lookup_inverse_found_time);
#endif

  if (first) {
    lengthExpected = newoff;
    first = false;
  }
  else if (lengthExpected != newoff) {
    printf("noutput expected %d actual %d\n", lengthExpected, newoff);
    exit(-1);
  }
}

void hashItTBB()
{
  // Step 1
  double t = omp_get_wtime();
  tbb::concurrent_unordered_map<int, int> col_map_offd_inverse(num_cols_A_offd);
#pragma omp parallel for
  for (int i = 0; i < num_cols_A_offd; ++i) {
    col_map_offd_inverse[col_map_offd[i]] = i;
  }

  double inverse_col_map_offd_time = omp_get_wtime() - t; t = omp_get_wtime();

  // Step 2
  int prefix_sum_workspace[omp_get_max_threads() + 1];
  int newoff;

  tbb::concurrent_unordered_set<int> found_set;
  double create_hash_time = omp_get_wtime() - t; t = omp_get_wtime();

  int map_lookup_cnt = 0, set_lookup_cnt = 0, set_insert_cnt = 0;

#ifdef CNT
#pragma omp parallel for reduction(+:map_lookup_cnt,set_lookup_cnt,set_insert_cnt)
#else
#pragma omp parallel for
#endif
  for (int i = 0; i < num_cols_A_offd; ++i) {
    if (CF_marker_offd[i] < 0) {
      for (int j = A_ext_i[i]; j < A_ext_i[i+1]; ++j) {
        int i1 = A_ext_j[j];
        if (i1 < col_1 || i1 >= col_n) {
#ifdef CNT
          ++set_lookup_cnt;
#endif
          if (found_set.find(i1) == found_set.end()) {
            auto itr = col_map_offd_inverse.find(i1);
#ifdef CNT
            ++map_lookup_cnt;
#endif
            if (itr == col_map_offd_inverse.end()) {
              found_set.insert(i1);
#ifdef CNT
              ++set_insert_cnt;
#endif
            }
            else {
              A_ext_j[j] = -itr->second - 1;
            }
          }
        }
      }
      for (int j = Sop_i[i]; j < Sop_i[i+1]; ++j) {
        int i1 = Sop_j[j];
        if (i1 < col_1 || i1 >= col_n) {
#ifdef CNT
          ++set_lookup_cnt;
#endif
          if (found_set.find(i1) == found_set.end()) {
            Sop_j[j] = -col_map_offd_inverse[i1] - 1;
#ifdef CNT
            ++map_lookup_cnt;
#endif
          }
        }
      }
    } // CF_marker_offd[i] < 0
  } // for each row

  double insert_hash_time = omp_get_wtime() - t; t = omp_get_wtime();

  // Step 3
  int i = 0;
  newoff = found_set.size();
  int *found = new int[found_set.size()];
  for (auto itr = found_set.begin(); itr != found_set.end(); ++itr, ++i) {
    found[i] = *itr;
  }

  double found_time = omp_get_wtime() - t; t = omp_get_wtime();

#ifdef VERBOSE
  printf("*Baseline: ninput: %d noutput: %d\n", A_ext_i[num_cols_A_offd], newoff);
#endif

  // Step 4
  int *temp = new int[newoff];
  int *sorted;
  merge_sort(found, temp, newoff, &sorted);
  if (sorted == found) {
    delete[] temp;
  }
  else {
    free(found);
  }

  double merge_time = omp_get_wtime() - t; t = omp_get_wtime();

  // Step 5
  tbb::concurrent_unordered_map<int, int> found_inverse(newoff);
#pragma omp parallel for
  for (int i = 0; i < newoff; ++i) {
    found_inverse[sorted[i]] = i;
  }
  found = sorted;

  double inverse_found_time = omp_get_wtime() - t; t = omp_get_wtime();

  // Step 6
  int lookup_inverse_cnt = 0;
#pragma omp parallel for
  for (int i = 0; i < num_cols_A_offd; ++i) {
    if (CF_marker_offd[i] < 0) {
      for (int kk = Sop_i[i]; kk < Sop_i[i + 1]; ++kk) {
        int k1 = Sop_j[kk];
        if (k1 > -1 && (k1 < col_1 || k1 >= col_n)) {
          int got_loc = found_inverse[k1];
          int loc_col = got_loc + num_cols_A_offd;
          Sop_j[kk] = -loc_col - 1;
#ifdef CNT
          ++lookup_inverse_cnt;
#endif
        }
      }
      for (int kk = A_ext_i[i]; kk < A_ext_i[i + 1]; ++kk) {
        int k1 = A_ext_j[kk];
        if (k1 > -1 && (k1 < col_1 || k1 >= col_n)) {
          int got_loc = found_inverse[k1];
          int loc_col = got_loc + num_cols_A_offd;
          A_ext_j[kk] = -loc_col - 1;
#ifdef CNT
          ++lookup_inverse_cnt;
#endif
        }
      }
    }
  }

  double lookup_inverse_found_time = omp_get_wtime() - t;

#ifdef VERBOSE
  printf(
    "inverse_col_map_offd takes %f (throughput %g)\n",
    inverse_col_map_offd_time, num_cols_A_offd/inverse_col_map_offd_time);
  printf(
    "create_hash takes %f (throughput %g)\n",
    create_hash_time, A_ext_i[num_cols_A_offd]/create_hash_time);
  printf(
    "insert_hash takes %f (throughput %g set_lookup %d set_insert %d map_lookup %d)\n",
    insert_hash_time,
    (set_lookup_cnt + set_insert_cnt + map_lookup_cnt)/insert_hash_time,
    set_lookup_cnt,
    set_insert_cnt,
    map_lookup_cnt);
  printf(
    "found takes %f (throughput %g)\n", found_time, A_ext_i[num_cols_A_offd]/found_time);
  printf(
    "merge takes %f (throughput %g)\n", merge_time, newoff/merge_time);
  printf(
    "inverse_found_time takes %f (throughput %g)\n", inverse_found_time, newoff/inverse_found_time);
  printf(
    "lookup_inverse_found_time takes %f (throughput %g)\n\n", lookup_inverse_found_time, lookup_inverse_cnt/lookup_inverse_found_time);
#endif

  if (first) {
    lengthExpected = newoff;
    first = false;
  }
  else if (lengthExpected != newoff) {
    printf("noutput expected %d actual %d\n", lengthExpected, newoff);
    exit(-1);
  }
}

void hashItHopscotch()
{
  // Step 1
  double t = omp_get_wtime();
  HopscotchUnorderedIntMap col_map_offd_inverse;
  hopscotchUnorderedIntMapCreate(&col_map_offd_inverse, 2*num_cols_A_offd, 16*omp_get_max_threads());
#pragma omp parallel for
  for (int i = 0; i < num_cols_A_offd; ++i) {
    int old = hopscotchUnorderedIntMapPutIfAbsent(&col_map_offd_inverse, col_map_offd[i], i);
  }
  double inverse_col_map_offd_time = omp_get_wtime() - t; t = omp_get_wtime();

  // Step 2
  int prefix_sum_workspace[omp_get_max_threads() + 1];
  int newoff;
  int *found;

  HopscotchUnorderedIntSet found_set;
  hopscotchUnorderedIntSetCreate(&found_set, A_ext_i[num_cols_A_offd], 16*omp_get_max_threads());
  double create_hash_time = omp_get_wtime() - t; t = omp_get_wtime();

  int map_lookup_cnt = 0, set_lookup_cnt = 0, set_insert_cnt = 0;

#ifdef CNT
#pragma omp parallel reduction(+:map_lookup_cnt,set_lookup_cnt,set_insert_cnt)
#else
#pragma omp parallel
#endif
  {
#pragma omp for
    for (int i = 0; i < num_cols_A_offd; ++i) {
      if (CF_marker_offd[i] < 0) {
        for (int j = A_ext_i[i]; j < A_ext_i[i+1]; ++j) {
          int i1 = A_ext_j[j];
          if (i1 < col_1 || i1 >= col_n) {
#ifdef CNT
            ++set_lookup_cnt;
#endif
            if (!hopscotchUnorderedIntSetContains(&found_set, i1)) {
              int data = hopscotchUnorderedIntMapGet(&col_map_offd_inverse, i1);
#ifdef CNT
              ++map_lookup_cnt;
#endif
              if (-1 == data) {
                hopscotchUnorderedIntSetPut(&found_set, i1);
#ifdef CNT
                ++set_insert_cnt;
#endif
              }
              else {
                A_ext_j[j] = -data - 1;
              }
            }
          }
        }
        for (int j = Sop_i[i]; j < Sop_i[i+1]; ++j) {
          int i1 = Sop_j[j];
          if (i1 < col_1 || i1 >= col_n) {
#ifdef CNT
            ++set_lookup_cnt;
#endif
            if (!hopscotchUnorderedIntSetContains(&found_set, i1)) {
              Sop_j[j] = -hopscotchUnorderedIntMapGet(&col_map_offd_inverse, i1) - 1;
#ifdef CNT
              ++map_lookup_cnt;
#endif
            }
          }
        }
      } // CF_marker_offd[i] < 0
    } // for each row
  } // omp parallel

  double insert_hash_time = omp_get_wtime() - t; t = omp_get_wtime();

  // Step 3
  found = hopscotchUnorderedIntSetCopyToArray(&found_set, &newoff);

  double found_time = omp_get_wtime() - t; t = omp_get_wtime();

#ifdef VERBOSE
  printf("*Hopscotch: ninput: %d noutput: %d\n", A_ext_i[num_cols_A_offd], newoff);
#endif

  // Step 4
  int *temp = new int[newoff];
  int *sorted;
  merge_sort(found, temp, newoff, &sorted);
  if (sorted == found) {
    delete[] temp;
  }
  else {
    free(found);
  }

  double merge_time = omp_get_wtime() - t; t = omp_get_wtime();

  // Step 5
  HopscotchUnorderedIntMap found_inverse;
  hopscotchUnorderedIntMapCreate(&found_inverse, newoff*2, 16*omp_get_max_threads());
#pragma omp parallel for
  for (int i = 0; i < newoff; ++i) {
    hopscotchUnorderedIntMapPutIfAbsent(&found_inverse, sorted[i], i);
  }
  found = sorted;

  double inverse_found_time = omp_get_wtime() - t; t = omp_get_wtime();

  // Step 6
  int lookup_inverse_cnt = 0;
#ifdef CNT
#pragma omp parallel for reduction(+:lookup_inverse_cnt)
#else
#pragma omp parallel for
#endif
  for (int i = 0; i < num_cols_A_offd; ++i) {
    if (CF_marker_offd[i] < 0) {
      for (int kk = Sop_i[i]; kk < Sop_i[i + 1]; ++kk) {
        int k1 = Sop_j[kk];
        if (k1 > -1 && (k1 < col_1 || k1 >= col_n)) {
          int got_loc = hopscotchUnorderedIntMapGet(&found_inverse, k1);
          int loc_col = got_loc + num_cols_A_offd;
          Sop_j[kk] = -loc_col - 1;
#ifdef CNT
          ++lookup_inverse_cnt;
#endif
        }
      }
      for (int kk = A_ext_i[i]; kk < A_ext_i[i + 1]; ++kk) {
        int k1 = A_ext_j[kk];
        if (k1 > -1 && (k1 < col_1 || k1 >= col_n)) {
          int got_loc = hopscotchUnorderedIntMapGet(&found_inverse, k1);
          int loc_col = got_loc + num_cols_A_offd;
          A_ext_j[kk] = -loc_col - 1;
#ifdef CNT
          ++lookup_inverse_cnt;
#endif
        }
      }
    }
  }

  double lookup_inverse_found_time = omp_get_wtime() - t;

#ifdef VERBOSE
  printf(
    "inverse_col_map_offd takes %f (throughput %g)\n",
    inverse_col_map_offd_time, num_cols_A_offd/inverse_col_map_offd_time);
  printf(
    "create_hash takes %f (throughput %g)\n",
    create_hash_time, A_ext_i[num_cols_A_offd]/create_hash_time);
  printf(
    "insert_hash takes %f (throughput %g set_lookup %d set_insert %d map_lookup %d)\n",
    insert_hash_time,
    (set_lookup_cnt + set_insert_cnt + map_lookup_cnt)/insert_hash_time,
    set_lookup_cnt,
    set_insert_cnt,
    map_lookup_cnt);
  printf(
    "found takes %f (throughput %g)\n", found_time, A_ext_i[num_cols_A_offd]/found_time);
  printf(
    "merge takes %f (throughput %g)\n", merge_time, newoff/merge_time);
  printf(
    "inverse_found_time takes %f (throughput %g)\n", inverse_found_time, newoff/inverse_found_time);
  printf(
    "lookup_inverse_found_time takes %f (throughput %g)\n\n", lookup_inverse_found_time, lookup_inverse_cnt/lookup_inverse_found_time);
#endif

  hopscotchUnorderedIntMapDestroy(&col_map_offd_inverse);
  hopscotchUnorderedIntMapDestroy(&found_inverse);

  if (lengthExpected != newoff) {
    printf("noutput expected %d actual %d\n", lengthExpected, newoff);
    exit(-1);
  }
}

void hashItPrivate()
{
  // Step 1
  double t = omp_get_wtime();
  HopscotchUnorderedIntMap col_map_offd_inverse;
  hopscotchUnorderedIntMapCreate(&col_map_offd_inverse, 2*num_cols_A_offd, 16*omp_get_max_threads());
#pragma omp parallel for
  for (int i = 0; i < num_cols_A_offd; ++i) {
    int old = hopscotchUnorderedIntMapPutIfAbsent(&col_map_offd_inverse, col_map_offd[i], i);
  }
  double inverse_col_map_offd_time = omp_get_wtime() - t; t = omp_get_wtime();

  // Steps 2 and 3
  int prefix_sum_workspace[omp_get_max_threads() + 1];
  int newoff;
  int *found;
  double insert_hash_time;

  int map_lookup_cnt = 0, set_lookup_cnt = 0, set_insert_cnt = 0;

#ifdef CNT
#pragma omp parallel reduction(+:map_lookup_cnt,set_lookup_cnt,set_insert_cnt)
#else
#pragma omp parallel
#endif
  {
    unordered_set<int> found_set;

#pragma omp for
    for (int i = 0; i < num_cols_A_offd; ++i) {
      if (CF_marker_offd[i] < 0) {
        for (int j = A_ext_i[i]; j < A_ext_i[i+1]; ++j) {
          int i1 = A_ext_j[j];
          if (i1 < col_1 || i1 >= col_n) {
#ifdef CNT
            ++set_lookup_cnt;
#endif
            if (found_set.find(i1) == found_set.end()) {
              int data = hopscotchUnorderedIntMapGet(&col_map_offd_inverse, i1);
#ifdef CNT
              ++map_lookup_cnt;
#endif
              if (-1 == data) {
                found_set.insert(i1);
#ifdef CNT
                ++set_insert_cnt;
#endif
              }
              else {
                //assert(col_map_offd_inverse2.find(i1) != col_map_offd_inverse2.end());
                //assert(data == col_map_offd_inverse2[i1]);
                A_ext_j[j] = -data - 1;
              }
            }
          }
        }
        for (int j = Sop_i[i]; j < Sop_i[i+1]; ++j) {
          int i1 = Sop_j[j];
          if (i1 < col_1 || i1 >= col_n) {
#ifdef CNT
            ++set_lookup_cnt;
#endif
            if (found_set.find(i1) == found_set.end()) {
              int data = hopscotchUnorderedIntMapGet(&col_map_offd_inverse, i1);
              assert(data != -1);
              //assert(data == col_map_offd_inverse2[i1]);
              Sop_j[j] = -data - 1;
#ifdef CNT
              ++map_lookup_cnt;
#endif
            }
          }
        }
      } // CF_marker_offd[i] < 0
    } // for each row

    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    prefix_sum_workspace[tid + 1] = found_set.size();

#pragma omp barrier
#pragma omp master
    {
      insert_hash_time = omp_get_wtime() - t; t = omp_get_wtime();

      prefix_sum_workspace[0] = 0;
      for (int i = 1; i < nthreads; ++i) {
        prefix_sum_workspace[i + 1] += prefix_sum_workspace[i];
      }
      newoff = prefix_sum_workspace[nthreads];
      found = new int[newoff];
    }
#pragma omp barrier

    int found_set_size = prefix_sum_workspace[tid];

    for (auto i : found_set) {
      found[found_set_size++] = i;
    }
  } // omp parallel

  double found_time = omp_get_wtime() - t; t = omp_get_wtime();

  // Steps 4 and 5
  HopscotchUnorderedIntMap found_inverse;
  int newoff_backup = newoff;
  newoff = sort_unique_and_inverse_map(found, newoff, &found, &found_inverse);

#ifdef VERBOSE
  printf("*Private: ninput: %d nthread-private-output: %d noutput: %d\n", A_ext_i[num_cols_A_offd], newoff_backup, newoff);
#endif

  double merge_inverse_found_time = omp_get_wtime() - t; t = omp_get_wtime();

  // Step 6
  int lookup_inverse_cnt = 0;
#ifdef CNT
#pragma omp parallel for reduction(+:lookup_inverse_cnt)
#else
#pragma omp parallel for
#endif
  for (int i = 0; i < num_cols_A_offd; ++i) {
    if (CF_marker_offd[i] < 0) {
      for (int kk = Sop_i[i]; kk < Sop_i[i + 1]; ++kk) {
        int k1 = Sop_j[kk];
        if (k1 > -1 && (k1 < col_1 || k1 >= col_n)) {
          int got_loc = hopscotchUnorderedIntMapGet(&found_inverse, k1);
          int loc_col = got_loc + num_cols_A_offd;
          Sop_j[kk] = -loc_col - 1;
#ifdef CNT
          ++lookup_inverse_cnt;
#endif
        }
      }
      for (int kk = A_ext_i[i]; kk < A_ext_i[i + 1]; ++kk) {
        int k1 = A_ext_j[kk];
        if (k1 > -1 && (k1 < col_1 || k1 >= col_n)) {
          int got_loc = hopscotchUnorderedIntMapGet(&found_inverse, k1);
          int loc_col = got_loc + num_cols_A_offd;
          A_ext_j[kk] = -loc_col - 1;
#ifdef CNT
          ++lookup_inverse_cnt;
#endif
        }
      }
    }
  }

  double lookup_inverse_found_time = omp_get_wtime() - t;

#ifdef VERBOSE
  printf(
    "inverse_col_map_offd takes %f (throughput %g)\n",
    inverse_col_map_offd_time, num_cols_A_offd/inverse_col_map_offd_time);
  //printf(
    //"insert_hash takes %f (throughput %g set_lookup %d set_insert %d map_lookup %d)\n",
    //insert_hash_time,
    //(set_lookup_cnt + set_insert_cnt + map_lookup_cnt)/insert_hash_time,
    //set_lookup_cnt,
    //set_insert_cnt,
    //map_lookup_cnt);
  printf(
    "found takes %f (throughput %g)\n", found_time, A_ext_i[num_cols_A_offd]/found_time);
  printf(
    "merge_inverse_found_time takes %f (throughput %g)\n", merge_inverse_found_time, newoff/merge_inverse_found_time);
  printf(
    "lookup_inverse_found_time takes %f (throughput %g)\n\n", lookup_inverse_found_time, lookup_inverse_cnt/lookup_inverse_found_time);
#endif

  hopscotchUnorderedIntMapDestroy(&col_map_offd_inverse);
  hopscotchUnorderedIntMapDestroy(&found_inverse);

  if (lengthExpected != newoff) {
    printf("noutput expected %d actual %d\n", lengthExpected, newoff);
    exit(-1);
  }
}

int main(int argc, char *argv[])
{
  double sumBaseline = 0, minBaseline = DBL_MAX;
  double sumCritical = 0, minCritical = DBL_MAX;
  double sumTBB = 0, minTBB = DBL_MAX;
  double sumPrivate = 0, minPrivate = DBL_MAX;
  double sumSerialHopscotch = 0, minSerialHopscotch = DBL_MAX;
  double sumHopscotch = 0, minHopscotch = DBL_MAX;

  if (argc < 2) {
    fprintf(stderr, "Usage: %s in_file_name\n", argv[0]);
    return -1;
  }

  FILE *fp = fopen(argv[1], "r");
  fread(&col_1, sizeof(col_1), 1, fp);
  fread(&col_n, sizeof(col_n), 1, fp);
  fread(&num_cols_A_offd, sizeof(num_cols_A_offd), 1, fp);

  col_map_offd = new int[num_cols_A_offd];
  CF_marker_offd = new int[num_cols_A_offd];
  A_ext_i = new int[num_cols_A_offd + 1];
  Sop_i = new int[num_cols_A_offd + 1];
  fread(col_map_offd, sizeof(col_map_offd[0]), num_cols_A_offd, fp);
  fread(CF_marker_offd, sizeof(CF_marker_offd[0]), num_cols_A_offd, fp);
  fread(A_ext_i, sizeof(A_ext_i[0]), num_cols_A_offd + 1, fp);

  A_ext_j = new int[A_ext_i[num_cols_A_offd]];
  int *A_ext_j_backup = new int[A_ext_i[num_cols_A_offd]];
  fread(A_ext_j, sizeof(A_ext_j[0]), A_ext_i[num_cols_A_offd], fp);
  memcpy(A_ext_j_backup, A_ext_j, sizeof(A_ext_j[0])*A_ext_i[num_cols_A_offd]);

  fread(Sop_i, sizeof(Sop_i[0]), num_cols_A_offd + 1, fp);
  Sop_j = new int[Sop_i[num_cols_A_offd]];
  int *Sop_j_backup = new int[Sop_i[num_cols_A_offd]];
  fread(Sop_j, sizeof(Sop_j[0]), Sop_i[num_cols_A_offd], fp);
  memcpy(Sop_j_backup, Sop_j, sizeof(Sop_j[0])*Sop_i[num_cols_A_offd]);

  fclose(fp);

  const int ITER = 16;
  for (int i = 0; i < ITER; i++) {
    double t = omp_get_wtime();
    hashItBaseline();
    double dt = omp_get_wtime() - t;
    sumBaseline += dt;
    minBaseline = std::min(dt, minBaseline);

    memcpy(A_ext_j, A_ext_j_backup, sizeof(A_ext_j[0])*A_ext_i[num_cols_A_offd]);
    memcpy(Sop_j, Sop_j_backup, sizeof(Sop_j[0])*Sop_i[num_cols_A_offd]);

    /*double tCritical = hashItCritical(v);
    sumCritical += tCritical;
    minCritical = std::min(tCritical, minCritical);*/

    /*double tTBB = hashItTBB(v);
    sumTBB += tTBB;
    minTBB = std::min(tTBB, minTBB);*/

    /*int old_nthreads = omp_get_max_threads();
    omp_set_num_threads(1);
    t = omp_get_wtime();
    hashItHopscotch();
    dt = omp_get_wtime() - t;
    sumSerialHopscotch += dt;
    minSerialHopscotch = std::min(dt, minSerialHopscotch);
    omp_set_num_threads(old_nthreads);

    memcpy(A_ext_j, A_ext_j_backup, sizeof(A_ext_j[0])*A_ext_i[num_cols_A_offd]);
    memcpy(Sop_j, Sop_j_backup, sizeof(Sop_j[0])*Sop_i[num_cols_A_offd]);*/

    t = omp_get_wtime();
    hashItTBB();
    dt = omp_get_wtime() - t;
    sumTBB += dt;
    minTBB = std::min(dt, minTBB);

    memcpy(A_ext_j, A_ext_j_backup, sizeof(A_ext_j[0])*A_ext_i[num_cols_A_offd]);
    memcpy(Sop_j, Sop_j_backup, sizeof(Sop_j[0])*Sop_i[num_cols_A_offd]);

    t = omp_get_wtime();
    hashItHopscotch();
    dt = omp_get_wtime() - t;
    sumHopscotch += dt;
    minHopscotch = std::min(dt, minHopscotch);

    memcpy(A_ext_j, A_ext_j_backup, sizeof(A_ext_j[0])*A_ext_i[num_cols_A_offd]);
    memcpy(Sop_j, Sop_j_backup, sizeof(Sop_j[0])*Sop_i[num_cols_A_offd]);

    t = omp_get_wtime();
    hashItPrivate();
    dt = omp_get_wtime() - t;
    sumPrivate += dt;
    minPrivate = std::min(dt, minPrivate);

    memcpy(A_ext_j, A_ext_j_backup, sizeof(A_ext_j[0])*A_ext_i[num_cols_A_offd]);
    memcpy(Sop_j, Sop_j_backup, sizeof(Sop_j[0])*Sop_i[num_cols_A_offd]);

    //sumHopscotch += tHopscotch;
    //minHopscotch = std::min(tHopscotch, minHopscotch);
  }

  int n = A_ext_i[num_cols_A_offd];
  printf("c++ std:   avg %f max %f mop/s\n", n/(sumBaseline/ITER)/1e6, n/minBaseline/1e6);
  //printf("SerialHopscotch: avg %f max %f mop/s\n", n/(sumSerialHopscotch/ITER)/1e6, n/minSerialHopscotch/1e6);
  printf("TBB: avg %f max %f mop/s\n", n/(sumTBB/ITER)/1e6, n/minTBB/1e6);
  printf("Hopscotch: avg %f max %f mop/s\n", n/(sumHopscotch/ITER)/1e6, n/minHopscotch/1e6);
  //printf("Private:   avg %f max %f mop/s\n", n/(sumPrivate/ITER)/1e6, n/minPrivate/1e6);

  return 0;
}
