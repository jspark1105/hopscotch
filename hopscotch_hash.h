/**
 * Hopscotch hash is modified from the code downloaded from
 * https://sites.google.com/site/cconcurrencypackage/hopscotch-hashing
 * with the following terms of usage
 */

////////////////////////////////////////////////////////////////////////////////
//TERMS OF USAGE
//------------------------------------------------------------------------------
//
//  Permission to use, copy, modify and distribute this software and
//  its documentation for any purpose is hereby granted without fee,
//  provided that due acknowledgments to the authors are provided and
//  this permission notice appears in all copies of the software.
//  The software is provided "as is". There is no warranty of any kind.
//
//Authors:
//  Maurice Herlihy
//  Brown University
//  and
//  Nir Shavit
//  Tel-Aviv University
//  and
//  Moran Tzafrir
//  Tel-Aviv University
//
//  Date: July 15, 2008.  
//
////////////////////////////////////////////////////////////////////////////////
// Programmer : Moran Tzafrir (MoranTza@gmail.com)
// Modified   : Jongsoo Park  (jongsoo.park@intel.com)
//              Oct 1, 2015.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HOPSCOTCH_HASH_HEADER
#define HOPSCOTCH_HASH_HEADER

#include <stdio.h>
#include <limits.h>
#include <assert.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define CONCURRENT_HOPSCOTCH

// Potentially architecture specific features used here:
// __builtin_ffs
// __sync_val_compare_and_swap

#ifdef __cplusplus
extern "C" {
#endif

#ifdef CONCURRENT_HOPSCOTCH
typedef struct {
  int volatile timestamp;
  omp_lock_t         lock;
} HopscotchSegment;
#endif

/**
 * The current typical use case of unordered set is putting input sequence
 * with lots of duplication (putting all colidx received from other ranks),
 * followed by one sweep of enumeration.
 * Since the capacity is set to the number of inputs, which is much larger
 * than the number of unique elements, we optimize for initialization and
 * enumeration whose time is proportional to the capacity.
 * For initialization and enumeration, structure of array (SoA) is better
 * for vectorization, cache line utilization, and so on.
 */
typedef struct
{
	int  volatile              segmentMask;
	int  volatile              bucketMask;
#ifdef CONCURRENT_HOPSCOTCH
	HopscotchSegment* volatile segments;
#endif
  int *volatile              key;
  unsigned int *volatile             hopInfo;
	int *volatile	             hash;
} HopscotchUnorderedIntSet;

typedef struct
{
  unsigned int volatile hopInfo;
  int  volatile hash;
  int  volatile key;
  int  volatile data;
} HopscotchBucket;

/**
 * The current typical use case of unoredered map is putting input sequence
 * with no duplication (inverse map of a bijective mapping) followed by
 * lots of lookups.
 * For lookup, array of structure (AoS) gives better cache line utilization.
 */
typedef struct
{
	int  volatile              segmentMask;
	int  volatile              bucketMask;
#ifdef CONCURRENT_HOPSCOTCH
	HopscotchSegment*	volatile segments;
#endif
	HopscotchBucket* volatile	 table;
} HopscotchUnorderedIntMap;

// Constants ................................................................
#define HOPSCOTCH_HASH_HOP_RANGE    (32)
#define HOPSCOTCH_HASH_INSERT_RANGE (4*1024)

#define HOPSCOTCH_HASH_EMPTY (0)
#define HOPSCOTCH_HASH_BUSY  (1)

// Small Utilities ..........................................................
static int firstLsbBitIndx(unsigned int x) {
  if (0 == x) return -1;
  return __builtin_ffs(x) - 1;
}

/**
 * HopscotchHash is adapted from xxHash with the following license.
 */
/*
   xxHash - Extremely Fast Hash algorithm
   Header File
   Copyright (C) 2012-2015, Yann Collet.

   BSD 2-Clause License (http://www.opensource.org/licenses/bsd-license.php)

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:

       * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
       * Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following disclaimer
   in the documentation and/or other materials provided with the
   distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   You can contact the author at :
   - xxHash source repository : https://github.com/Cyan4973/xxHash
*/

/***************************************
*  Constants
***************************************/
#define HOPSCOTCH_XXH_PRIME32_1   2654435761U
#define HOPSCOTCH_XXH_PRIME32_2   2246822519U
#define HOPSCOTCH_XXH_PRIME32_3   3266489917U
#define HOPSCOTCH_XXH_PRIME32_4    668265263U
#define HOPSCOTCH_XXH_PRIME32_5    374761393U

#define HOPSCOTCH_XXH_PRIME64_1 11400714785074694791ULL
#define HOPSCOTCH_XXH_PRIME64_2 14029467366897019727ULL
#define HOPSCOTCH_XXH_PRIME64_3  1609587929392839161ULL
#define HOPSCOTCH_XXH_PRIME64_4  9650029242287828579ULL
#define HOPSCOTCH_XXH_PRIME64_5  2870177450012600261ULL

#  define HOPSCOTCH_XXH_rotl32(x,r) ((x << r) | (x >> (32 - r)))
#  define HOPSCOTCH_XXH_rotl64(x,r) ((x << r) | (x >> (64 - r)))

static int hopscotchHash(int input)
{
    unsigned int h32 = HOPSCOTCH_XXH_PRIME32_5 + sizeof(input);

    // 1665863975 is added to input so that
    // only -1073741824 gives HOPSCOTCH_HASH_EMPTY.
    // Hence, we're fine as long as key is non-negative.
    h32 += (input + 1665863975)*HOPSCOTCH_XXH_PRIME32_3;
    h32 = HOPSCOTCH_XXH_rotl32(h32, 17)*HOPSCOTCH_XXH_PRIME32_4;

    h32 ^= h32 >> 15;
    h32 *= HOPSCOTCH_XXH_PRIME32_2;
    h32 ^= h32 >> 13;
    h32 *= HOPSCOTCH_XXH_PRIME32_3;
    h32 ^= h32 >> 16;

    //assert(HOPSCOTCH_HASH_EMPTY != h32);

    return h32;
}

static void hopscotchUnorderedIntSetFindCloserFreeBucket(HopscotchUnorderedIntSet *s,
#ifdef CONCURRENT_HOPSCOTCH
                                                         HopscotchSegment* start_seg,
#endif
                                                         int *free_bucket,
                                                         int *free_dist )
{
  int move_bucket = *free_bucket - (HOPSCOTCH_HASH_HOP_RANGE - 1);
  int move_free_dist;
  for (move_free_dist = HOPSCOTCH_HASH_HOP_RANGE - 1; move_free_dist > 0; --move_free_dist)
  {
    unsigned int start_hop_info = s->hopInfo[move_bucket];
    int move_new_free_dist = -1;
    unsigned int mask = 1;
    int i;
    for (i = 0; i < move_free_dist; ++i, mask <<= 1)
    {
      if (mask & start_hop_info)
      {
        move_new_free_dist = i;
        break;
      }
    }
    if (-1 != move_new_free_dist)
    {
#ifdef CONCURRENT_HOPSCOTCH
      HopscotchSegment*  move_segment = &(s->segments[move_bucket & s->segmentMask]);
      
      if(start_seg != move_segment)
        omp_set_lock(&move_segment->lock);
#endif

      if (start_hop_info == s->hopInfo[move_bucket])
      {
        // new_free_bucket -> free_bucket and empty new_free_bucket
        int new_free_bucket = move_bucket + move_new_free_dist;
        s->key[*free_bucket]  = s->key[new_free_bucket];
        s->hash[*free_bucket] = s->hash[new_free_bucket];

#ifdef CONCURRENT_HOPSCOTCH
        ++move_segment->timestamp;
#pragma omp flush
#endif

        s->hopInfo[move_bucket] |= (1U << move_free_dist);
        s->hopInfo[move_bucket] &= ~(1U << move_new_free_dist);

        *free_bucket = new_free_bucket;
        *free_dist -= move_free_dist - move_new_free_dist;

#ifdef CONCURRENT_HOPSCOTCH
        if(start_seg != move_segment)
          omp_unset_lock(&move_segment->lock);
#endif

        return;
      }
#ifdef CONCURRENT_HOPSCOTCH
      if(start_seg != move_segment)
        omp_unset_lock(&move_segment->lock);
#endif
    }
    ++move_bucket;
  }
  *free_bucket = -1; 
  *free_dist = 0;
}

static void hopscotchUnorderedIntMapFindCloserFreeBucket(HopscotchUnorderedIntMap *m,
#ifdef CONCURRENT_HOPSCOTCH
                                                         HopscotchSegment* start_seg,
#endif
                                                         HopscotchBucket** free_bucket,
                                                         int* free_dist)
{
  HopscotchBucket* move_bucket = *free_bucket - (HOPSCOTCH_HASH_HOP_RANGE - 1);
  int move_free_dist;
  for (move_free_dist = HOPSCOTCH_HASH_HOP_RANGE - 1; move_free_dist > 0; --move_free_dist)
  {
    unsigned int start_hop_info = move_bucket->hopInfo;
    int move_new_free_dist = -1;
    unsigned int mask = 1;
    int i;
    for (i = 0; i < move_free_dist; ++i, mask <<= 1)
    {
      if (mask & start_hop_info)
      {
        move_new_free_dist = i;
        break;
      }
    }
    if (-1 != move_new_free_dist)
    {
#ifdef CONCURRENT_HOPSCOTCH
      HopscotchSegment* move_segment = &(m->segments[(move_bucket - m->table) & m->segmentMask]);
      
      if (start_seg != move_segment)
        omp_set_lock(&move_segment->lock);
#endif

      if (start_hop_info == move_bucket->hopInfo)
      {
        // new_free_bucket -> free_bucket and empty new_free_bucket
        HopscotchBucket* new_free_bucket = move_bucket + move_new_free_dist;
        (*free_bucket)->data = new_free_bucket->data;
        (*free_bucket)->key  = new_free_bucket->key;
        (*free_bucket)->hash = new_free_bucket->hash;

#ifdef CONCURRENT_HOPSCOTCH
        ++move_segment->timestamp;

#pragma omp flush
#endif

        move_bucket->hopInfo |= (1U << move_free_dist);
        move_bucket->hopInfo &= ~(1U << move_new_free_dist);

        *free_bucket = new_free_bucket;
        *free_dist -= move_free_dist - move_new_free_dist;

#ifdef CONCURRENT_HOPSCOTCH
        if(start_seg != move_segment)
          omp_unset_lock(&move_segment->lock);
#endif
        return;
      }
#ifdef CONCURRENT_HOPSCOTCH
      if(start_seg != move_segment)
        omp_unset_lock(&move_segment->lock);
#endif
    }
    ++move_bucket;
  }
  *free_bucket = NULL; 
  *free_dist = 0;
}

static void hopscotchUnorderedIntMapFindCloserFreeBucketNoSync(HopscotchUnorderedIntMap *m,
                                                               HopscotchBucket** free_bucket,
                                                               int* free_dist)
{
  HopscotchBucket* move_bucket = *free_bucket - (HOPSCOTCH_HASH_HOP_RANGE - 1);
  int move_free_dist;
  for (move_free_dist = HOPSCOTCH_HASH_HOP_RANGE - 1; move_free_dist > 0; --move_free_dist)
  {
    unsigned int start_hop_info = move_bucket->hopInfo;
    int move_new_free_dist = -1;
    unsigned int mask = 1;
    int i;
    for (i = 0; i < move_free_dist; ++i, mask <<= 1)
    {
      if (mask & start_hop_info)
      {
        move_new_free_dist = i;
        break;
      }
    }
    if (-1 != move_new_free_dist)
    {
      if (start_hop_info == move_bucket->hopInfo)
      {
        // new_free_bucket -> free_bucket and empty new_free_bucket
        HopscotchBucket* new_free_bucket = move_bucket + move_new_free_dist;
        (*free_bucket)->data = new_free_bucket->data;
        (*free_bucket)->key  = new_free_bucket->key;
        (*free_bucket)->hash = new_free_bucket->hash;

        move_bucket->hopInfo |= (1U << move_free_dist);
        move_bucket->hopInfo &= ~(1U << move_new_free_dist);

        *free_bucket = new_free_bucket;
        *free_dist -= move_free_dist - move_new_free_dist;
        return;
      }
    }
    ++move_bucket;
  }
  *free_bucket = NULL; 
  *free_dist = 0;
}


void hopscotchUnorderedIntSetCreate( HopscotchUnorderedIntSet *s,
                                     int inCapacity,
                                     int concurrencyLevel);
void hopscotchUnorderedIntMapCreate( HopscotchUnorderedIntMap *m,
                                     int inCapacity,
                                     int concurrencyLevel);

void hopscotchUnorderedIntSetDestroy( HopscotchUnorderedIntSet *s );
void hopscotchUnorderedIntMapDestroy( HopscotchUnorderedIntMap *m );

// Query Operations .........................................................
static int hopscotchUnorderedIntSetContains( HopscotchUnorderedIntSet *s,
                                             int key )
{
  //CALCULATE HASH ..........................
  int hash = hopscotchHash(key);

  //CHECK IF ALREADY CONTAIN ................
#ifdef CONCURRENT_HOPSCOTCH
  HopscotchSegment *segment = &s->segments[hash & s->segmentMask];
#endif
  int bucket = hash & s->bucketMask;
  unsigned int hopInfo = s->hopInfo[bucket];

  if (0 ==hopInfo)
    return 0;
  else if (1 == hopInfo )
  {
    if (hash == s->hash[bucket] && key == s->key[bucket])
      return 1;
    else return 0;
  }

#ifdef CONCURRENT_HOPSCOTCH
  int startTimestamp = segment->timestamp;
#endif
  while (0 != hopInfo)
  {
    int i = firstLsbBitIndx(hopInfo);
    int currElm = bucket + i;

    if (hash == s->hash[currElm] && key == s->key[currElm])
      return 1;
    hopInfo &= ~(1U << i);
  } 

#ifdef CONCURRENT_HOPSCOTCH
  if (segment->timestamp == startTimestamp)
    return 0;
#endif

  int i;
  for (i = 0; i< HOPSCOTCH_HASH_HOP_RANGE; ++i)
  {
    if (hash == s->hash[bucket + i] && key == s->key[bucket + i])
      return 1;
  }
  return 0;
}

/**
 * @ret -1 if key doesn't exist
 */
static int hopscotchUnorderedIntMapGet( HopscotchUnorderedIntMap *m,
                                        int key)
{
  //CALCULATE HASH ..........................
  int hash = hopscotchHash(key);

  //CHECK IF ALREADY CONTAIN ................
#ifdef CONCURRENT_HOPSCOTCH
  HopscotchSegment *segment = &m->segments[hash & m->segmentMask];
#endif
  HopscotchBucket *elmAry = &(m->table[hash & m->bucketMask]);
  unsigned int hopInfo = elmAry->hopInfo;
  if (0 == hopInfo)
    return -1;
  else if (1 == hopInfo )
  {
    if (hash == elmAry->hash && key == elmAry->key)
      return elmAry->data;
    else return -1;
  }

#ifdef CONCURRENT_HOPSCOTCH
  int startTimestamp = segment->timestamp;
#endif
  while (0 != hopInfo)
  {
    int i = firstLsbBitIndx(hopInfo);
    HopscotchBucket* currElm = elmAry + i;
    if (hash == currElm->hash && key == currElm->key)
      return currElm->data;
    hopInfo &= ~(1U << i);
  } 

#ifdef CONCURRENT_HOPSCOTCH
  if (segment->timestamp == startTimestamp)
    return -1;
#endif

  HopscotchBucket *currBucket = &(m->table[hash & m->bucketMask]);
  int i;
  for (i = 0; i< HOPSCOTCH_HASH_HOP_RANGE; ++i, ++currBucket)
  {
    if (hash == currBucket->hash && key == currBucket->key)
      return currBucket->data;
  }
  return -1;
}

static int hopscotchUnorderedIntMapGetNoSync( HopscotchUnorderedIntMap *m,
                                              int key)
{
  //CALCULATE HASH ..........................
  int hash = hopscotchHash(key);

  //CHECK IF ALREADY CONTAIN ................
  HopscotchBucket *elmAry = &(m->table[hash & m->bucketMask]);
  unsigned int hopInfo = elmAry->hopInfo;
  if (0 == hopInfo)
    return -1;
  else if (1 == hopInfo )
  {
    if (hash == elmAry->hash && key == elmAry->key)
      return elmAry->data;
    else return -1;
  }

  while (0 != hopInfo)
  {
    int i = firstLsbBitIndx(hopInfo);
    HopscotchBucket* currElm = elmAry + i;
    if (hash == currElm->hash && key == currElm->key)
      return currElm->data;
    hopInfo &= ~(1U << i);
  } 

  HopscotchBucket *currBucket = &(m->table[hash & m->bucketMask]);
  int i;
  for (i = 0; i< HOPSCOTCH_HASH_HOP_RANGE; ++i, ++currBucket)
  {
    if (hash == currBucket->hash && key == currBucket->key)
      return currBucket->data;
  }
  return -1;
}


//status Operations .........................................................
static int hopscotchUnorderedIntSetSize(HopscotchUnorderedIntSet *s)
{
  int counter = 0;
  int n = s->bucketMask + HOPSCOTCH_HASH_INSERT_RANGE;
  int i;
  for (i = 0; i < n; ++i)
  {
    if (HOPSCOTCH_HASH_EMPTY != s->hash[i])
    {
      ++counter;
    }
  }
  return counter;
}   

static int hopscotchUnorderedIntMapSize(HopscotchUnorderedIntMap *m)
{
  int counter = 0;
  int n = m->bucketMask + HOPSCOTCH_HASH_INSERT_RANGE;
  int i;
  for (i = 0; i < n; ++i)
  {
    if( HOPSCOTCH_HASH_EMPTY != m->table[i].hash )
    {
      ++counter;
    }
  }
  return counter;
}

int *hopscotchUnorderedIntSetCopyToArray( HopscotchUnorderedIntSet *s, int *len );

//modification Operations ...................................................
static void hopscotchUnorderedIntSetPut( HopscotchUnorderedIntSet *s,
                                         int key )
{
  //CALCULATE HASH ..........................
  int hash = hopscotchHash(key);

  //LOCK KEY HASH ENTERY ....................
#ifdef CONCURRENT_HOPSCOTCH
  HopscotchSegment  *segment = &s->segments[hash & s->segmentMask];
  omp_set_lock(&segment->lock);
#endif
  int bucket = hash&s->bucketMask;

  //CHECK IF ALREADY CONTAIN ................
  unsigned int hopInfo = s->hopInfo[bucket];
  while (0 != hopInfo)
  {
    int i = firstLsbBitIndx(hopInfo);
    int currElm = bucket + i;

    if(hash == s->hash[currElm] && key == s->key[currElm])
    {
#ifdef CONCURRENT_HOPSCOTCH
      omp_unset_lock(&segment->lock);
#endif
      return;
    }
    hopInfo &= ~(1U << i);
  }

  //LOOK FOR FREE BUCKET ....................
  int free_bucket = bucket;
  int free_dist = 0;
  for ( ; free_dist < HOPSCOTCH_HASH_INSERT_RANGE; ++free_dist, ++free_bucket)
  {
#ifdef CONCURRENT_HOPSCOTCH
    if( (HOPSCOTCH_HASH_EMPTY == s->hash[free_bucket]) && (HOPSCOTCH_HASH_EMPTY == __sync_val_compare_and_swap((int *)&s->hash[free_bucket], (int)HOPSCOTCH_HASH_EMPTY, (int)HOPSCOTCH_HASH_BUSY)) )
#else
    if( (HOPSCOTCH_HASH_EMPTY == s->hash[free_bucket]) )
#endif
      break;
  }

  //PLACE THE NEW KEY .......................
  if (free_dist < HOPSCOTCH_HASH_INSERT_RANGE)
  {
    do
    {
      if (free_dist < HOPSCOTCH_HASH_HOP_RANGE)
      {
        s->key[free_bucket]  = key;
        s->hash[free_bucket] = hash;
        s->hopInfo[bucket]  |= 1U << free_dist;

#ifdef CONCURRENT_HOPSCOTCH
        omp_unset_lock(&segment->lock);
#endif
        return;
      }
      hopscotchUnorderedIntSetFindCloserFreeBucket(s,
#ifdef CONCURRENT_HOPSCOTCH
                                                   segment,
#endif
                                                   &free_bucket, &free_dist);
    } while (-1 != free_bucket);
  }

  //NEED TO RESIZE ..........................
  fprintf(stderr, "ERROR - RESIZE is not implemented\n");
  exit(1);
  return;
}

static int hopscotchUnorderedIntMapPutIfAbsent( HopscotchUnorderedIntMap *m, int key, int data)
{
  //CALCULATE HASH ..........................
  int hash = hopscotchHash(key);

  //LOCK KEY HASH ENTERY ....................
#ifdef CONCURRENT_HOPSCOTCH
  HopscotchSegment *segment = &m->segments[hash & m->segmentMask];
  omp_set_lock(&segment->lock);
#endif
  HopscotchBucket* startBucket = &(m->table[hash & m->bucketMask]);

  //CHECK IF ALREADY CONTAIN ................
  unsigned int hopInfo = startBucket->hopInfo;
  while (0 != hopInfo)
  {
    int i = firstLsbBitIndx(hopInfo);
    HopscotchBucket* currElm = startBucket + i;
    if (hash == currElm->hash && key == currElm->key)
    {
      int rc = currElm->data;
#ifdef CONCURRENT_HOPSCOTCH
      omp_unset_lock(&segment->lock);
#endif
      return rc;
    }
    hopInfo &= ~(1U << i);
  }

  //LOOK FOR FREE BUCKET ....................
  HopscotchBucket* free_bucket = startBucket;
  int free_dist = 0;
  for ( ; free_dist < HOPSCOTCH_HASH_INSERT_RANGE; ++free_dist, ++free_bucket)
  {
#ifdef CONCURRENT_HOPSCOTCH
    if( (HOPSCOTCH_HASH_EMPTY == free_bucket->hash) && (HOPSCOTCH_HASH_EMPTY == __sync_val_compare_and_swap((int *)&free_bucket->hash, (int)HOPSCOTCH_HASH_EMPTY, (int)HOPSCOTCH_HASH_BUSY)) )
#else
    if( (HOPSCOTCH_HASH_EMPTY == free_bucket->hash) )
#endif
      break;
  }

  //PLACE THE NEW KEY .......................
  if (free_dist < HOPSCOTCH_HASH_INSERT_RANGE)
  {
    do
    {
      if (free_dist < HOPSCOTCH_HASH_HOP_RANGE)
      {
        free_bucket->data     = data;
        free_bucket->key      = key;
        free_bucket->hash     = hash;
        startBucket->hopInfo |= 1U << free_dist;
#ifdef CONCURRENT_HOPSCOTCH
        omp_unset_lock(&segment->lock);
#endif
        return HOPSCOTCH_HASH_EMPTY;
      }
      hopscotchUnorderedIntMapFindCloserFreeBucket(m,
#ifdef CONCURRENT_HOPSCOTCH
                                                segment,
#endif
                                                &free_bucket, &free_dist);
    } while (NULL != free_bucket);
  }

  //NEED TO RESIZE ..........................
  fprintf(stderr, "ERROR - RESIZE is not implemented\n");
  exit(1);
  return HOPSCOTCH_HASH_EMPTY;
}

static int hopscotchUnorderedIntMapPutIfAbsentNoSync( HopscotchUnorderedIntMap *m, int key, int data)
{
  //CALCULATE HASH ..........................
  int hash = hopscotchHash(key);

  //LOCK KEY HASH ENTERY ....................
  HopscotchBucket* startBucket = &(m->table[hash & m->bucketMask]);

  //CHECK IF ALREADY CONTAIN ................
  unsigned int hopInfo = startBucket->hopInfo;
  while (0 != hopInfo)
  {
    int i = firstLsbBitIndx(hopInfo);
    HopscotchBucket* currElm = startBucket + i;
    if (hash == currElm->hash && key == currElm->key)
    {
      int rc = currElm->data;
      return rc;
    }
    hopInfo &= ~(1U << i);
  }

  //LOOK FOR FREE BUCKET ....................
  HopscotchBucket* free_bucket = startBucket;
  int free_dist = 0;
  for ( ; free_dist < HOPSCOTCH_HASH_INSERT_RANGE; ++free_dist, ++free_bucket)
  {
    if( (HOPSCOTCH_HASH_EMPTY == free_bucket->hash) )
      break;
  }

  //PLACE THE NEW KEY .......................
  if (free_dist < HOPSCOTCH_HASH_INSERT_RANGE)
  {
    do
    {
      if (free_dist < HOPSCOTCH_HASH_HOP_RANGE)
      {
        free_bucket->data     = data;
        free_bucket->key      = key;
        free_bucket->hash     = hash;
        startBucket->hopInfo |= 1U << free_dist;
        return HOPSCOTCH_HASH_EMPTY;
      }
      hopscotchUnorderedIntMapFindCloserFreeBucketNoSync(m,
                                                      &free_bucket, &free_dist);
    } while (NULL != free_bucket);
  }

  //NEED TO RESIZE ..........................
  fprintf(stderr, "ERROR - RESIZE is not implemented\n");
  exit(1);
  return HOPSCOTCH_HASH_EMPTY;
}


#ifdef __cplusplus
} // extern "C"
#endif

#endif // HOPSCOTCH_HASH_HEADER
