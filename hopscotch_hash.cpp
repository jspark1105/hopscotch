#include <math.h>

#include "hopscotch_hash.h"

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

static void getSimpleThreadPartition(int* begin, int *end, int n)
{
  int nthreads = omp_get_num_threads();
  int tid = omp_get_thread_num();

  int n_per_thread = (n + nthreads - 1)/nthreads;

  *begin = MIN(n_per_thread*tid, n);
  *end = MIN(*begin + n_per_thread, n);
}

static int nearestPowerOfTwo( int value )
{
  int rc = 1;
  while (rc < value) {
    rc <<= 1;
  }
  return rc;
}

static void initBucket(HopscotchBucket *b)
{
  b->hopInfo = 0;
  b->hash = HOPSCOTCH_HASH_EMPTY;
}

#ifdef CONCURRENT_HOPSCOTCH
static void initSegment(HopscotchSegment *s)
{
  s->timestamp = 0;
  omp_init_lock(&s->lock);
}

static void destroySegment(HopscotchSegment *s)
{
  omp_destroy_lock(&s->lock);
}
#endif

void hopscotchUnorderedIntSetCreate( HopscotchUnorderedIntSet *s,
                                     int inCapacity,
                                     int concurrencyLevel) 
{
  s->segmentMask = nearestPowerOfTwo(concurrencyLevel) - 1;
  if (inCapacity < s->segmentMask + 1)
  {
    inCapacity = s->segmentMask + 1;
  }

  //ADJUST INPUT ............................
  int adjInitCap = nearestPowerOfTwo(inCapacity + 4096);
  int num_buckets = adjInitCap + HOPSCOTCH_HASH_INSERT_RANGE + 1;
  s->bucketMask = adjInitCap - 1;

  int i;

  //ALLOCATE THE SEGMENTS ...................
#ifdef CONCURRENT_HOPSCOTCH
  s->segments = (HopscotchSegment *)malloc(sizeof(HopscotchSegment)*(s->segmentMask + 1));
  for (i = 0; i <= s->segmentMask; ++i)
  {
    initSegment(&s->segments[i]);
  }
#endif

  s->hopInfo = (unsigned int *)malloc(sizeof(unsigned int)*num_buckets);
  s->key = (int *)malloc(sizeof(int)*num_buckets);
  s->hash = (int *)malloc(sizeof(int)*num_buckets);

#ifdef CONCURRENT_HOPSCOTCH
#pragma omp parallel for if(!omp_in_parallel())
#endif
  for (i = 0; i < num_buckets; ++i)
  {
    s->hopInfo[i] = 0;
    s->hash[i] = HOPSCOTCH_HASH_EMPTY;
  }
}

void hopscotchUnorderedIntMapCreate( HopscotchUnorderedIntMap *m,
                                     int inCapacity,
                                     int concurrencyLevel) 
{
  m->segmentMask = nearestPowerOfTwo(concurrencyLevel) - 1;
  if (inCapacity < m->segmentMask + 1)
  {
    inCapacity = m->segmentMask + 1;
  }

  //ADJUST INPUT ............................
  int adjInitCap = nearestPowerOfTwo(inCapacity + 4096);
  int num_buckets = adjInitCap + HOPSCOTCH_HASH_INSERT_RANGE + 1;
  m->bucketMask = adjInitCap - 1;

  int i;

  //ALLOCATE THE SEGMENTS ...................
#ifdef CONCURRENT_HOPSCOTCH
  m->segments = (HopscotchSegment *)malloc(sizeof(HopscotchSegment)*(m->segmentMask + 1));
  for (i = 0; i <= m->segmentMask; i++)
  {
    initSegment(&m->segments[i]);
  }
#endif

  m->table = (HopscotchBucket *)malloc(sizeof(HopscotchBucket)*num_buckets);

#ifdef CONCURRENT_HOPSCOTCH
#pragma omp parallel for
#endif
  for (i = 0; i < num_buckets; i++)
  {
    initBucket(&m->table[i]);
  }
}

void hopscotchUnorderedIntSetDestroy( HopscotchUnorderedIntSet *s )
{
  free(s->hopInfo);
  free(s->key);
  free(s->hash);

#ifdef CONCURRENT_HOPSCOTCH
  int i;
  for (i = 0; i <= s->segmentMask; i++)
  {
    destroySegment(&s->segments[i]);
  }
  free(s->segments);
#endif
}

void hopscotchUnorderedIntMapDestroy( HopscotchUnorderedIntMap *m)
{
  free(m->table);

#ifdef CONCURRENT_HOPSCOTCH
  int i;
  for (i = 0; i <= m->segmentMask; i++)
  {
    destroySegment(&m->segments[i]);
  }
  free(m->segments);
#endif
}

void prefixSum(int *in_out, int *sum, int *workspace)
{
  int nthreads = omp_get_num_threads();
  int tid = omp_get_thread_num();

  workspace[tid + 1] = *in_out;

#pragma omp barrier
#pragma omp master
  {
    workspace[0] = 0;
    int i;
    for (i = 1; i < nthreads; i++) {
      workspace[i + 1] += workspace[i];
    }
    *sum = workspace[nthreads];
  }
#pragma omp barrier

  *in_out = workspace[tid];
}

int *hopscotchUnorderedIntSetCopyToArray( HopscotchUnorderedIntSet *s, int *len )
{
  int prefix_sum_workspace[4096/*omp_get_num_threads() + 1*/];
  assert(omp_get_num_threads() < 4096);
  int *ret_array = NULL;

#ifdef CONCURRENT_HOPSCOTCH
#pragma omp parallel
#endif
  {
    int n = s->bucketMask + HOPSCOTCH_HASH_INSERT_RANGE;
    int i_begin, i_end;
    getSimpleThreadPartition(&i_begin, &i_end, n);

    int cnt = 0;
    int i;
    for (i = i_begin; i < i_end; i++)
    {
      if (HOPSCOTCH_HASH_EMPTY != s->hash[i]) cnt++;
    }

    prefixSum(&cnt, len, prefix_sum_workspace);

#ifdef CONCURRENT_HOPSCOTCH
#pragma omp barrier
#pragma omp master
#endif
    {
      ret_array = (int *)malloc(sizeof(int)*(*len));
    }
#ifdef CONCURRENT_HOPSCOTCH
#pragma omp barrier
#endif

    for (i = i_begin; i < i_end; i++)
    {
      if (HOPSCOTCH_HASH_EMPTY != s->hash[i]) ret_array[cnt++] = s->key[i];
    }
  }

  return ret_array;
}
