CXX = icpc
CFLAGS = -fopenmp -Wall -Wextra
ifeq (yes, $(DBG))
  CFLAGS += -g -O0
else
  CFLAGS += -O3 -DNDEBUG
endif
CFLAGS += -xHost
CXXFLAGS = ${CFLAGS} -tbb -std=c++11 -I..

hash_bench: main.cpp ../hopscotch_hash.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm hash_bench
