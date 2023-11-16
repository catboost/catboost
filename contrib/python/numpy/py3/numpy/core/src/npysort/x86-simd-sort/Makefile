CXX		?= g++-12
CXXFLAGS	+= -I$(SRCDIR) -I$(UTILS) -O3
GTESTCFLAGS	= `pkg-config --cflags gtest_main`
GTESTLDFLAGS	= `pkg-config --static --libs gtest_main`
GBENCHCFLAGS	= `pkg-config --cflags benchmark`
GBENCHLDFLAGS	= `pkg-config --static --libs benchmark`
MARCHFLAG	= -march=sapphirerapids

SRCDIR		= ./src
TESTDIR		= ./tests
BENCHDIR	= ./benchmarks
UTILS		= ./utils
SRCS		= $(wildcard $(SRCDIR)/*.hpp)
TESTS		= $(wildcard $(TESTDIR)/*.cpp)
BENCHS		= $(wildcard $(BENCHDIR)/*.cpp)
TESTOBJS	= $(patsubst $(TESTDIR)/%.cpp,$(TESTDIR)/%.o,$(TESTS))
BENCHOBJS	= $(patsubst $(BENCHDIR)/%.cpp,$(BENCHDIR)/%.o,$(BENCHS))

# Compiling AVX512-FP16 instructions isn't possible for g++ < 12
ifeq ($(shell expr `$(CXX) -dumpversion | cut -d '.' -f 1` \< 12), 1)
	MARCHFLAG = -march=icelake-client
	BENCHOBJS_SKIP += bench-qsortfp16.o
	TESTOBJS_SKIP += test-qsortfp16.o
endif

BENCHOBJS	:= $(filter-out $(addprefix $(BENCHDIR)/, $(BENCHOBJS_SKIP)) ,$(BENCHOBJS))
TESTOBJS	:= $(filter-out $(addprefix $(TESTDIR)/, $(TESTOBJS_SKIP)) ,$(TESTOBJS))

all : test bench

$(UTILS)/cpuinfo.o : $(UTILS)/cpuinfo.cpp
	$(CXX) $(CXXFLAGS) -c $(UTILS)/cpuinfo.cpp -o $(UTILS)/cpuinfo.o

$(TESTDIR)/%.o : $(TESTDIR)/%.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) $(MARCHFLAG) $(GTESTCFLAGS) -c $< -o $@

test: $(TESTOBJS) $(UTILS)/cpuinfo.o $(SRCS)
	$(CXX) $(TESTOBJS) $(UTILS)/cpuinfo.o $(MARCHFLAG) $(CXXFLAGS) -lgtest_main $(GTESTLDFLAGS) -o testexe

$(BENCHDIR)/%.o : $(BENCHDIR)/%.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) $(MARCHFLAG) $(GBENCHCFLAGS) -c $< -o $@

bench: $(BENCHOBJS) $(UTILS)/cpuinfo.o
	$(CXX) $(BENCHOBJS) $(UTILS)/cpuinfo.o $(MARCHFLAG) $(CXXFLAGS) -lbenchmark_main $(GBENCHLDFLAGS) -o benchexe

meson:
	meson setup --warnlevel 0 --buildtype plain builddir
	cd builddir && ninja

clean:
	$(RM) -rf $(TESTDIR)/*.o $(BENCHDIR)/*.o $(UTILS)/*.o testexe benchexe builddir
