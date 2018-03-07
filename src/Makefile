
# add parallelism equal to number of cores every time.
# it seems that adding -jX to MAKEFLAGS directly doesn't work any more.
# included some "random" strings to ensure uniqueness
ifneq ($(PARALELL_WRAPPER_ABXCOEOEKCOEBMQJKHTOEUB),done)

NUM_CORES ?= $(shell grep -c "vendor_id" /proc/cpuinfo)
MAKEFLAGS +=" -j$(NUM_CORES) -l$(NUM_CORES) "

# for the default target case
parallel_wrapper_default_target_anthsqjkshbeohcbmeuthnoethoaeou:
	@$(MAKE) PARALELL_WRAPPER_ABXCOEOEKCOEBMQJKHTOEUB=done

# catches everything else
% :
	@$(MAKE) $@ PARALELL_WRAPPER_ABXCOEOEKCOEBMQJKHTOEUB=done

# the match for this else is at the end of the file
else

.PHONY: all clean build_info

# remove ALL implicit rules & all suffixes
MAKEFLAGS+=" -r"

BUILD_MODE ?= release

BUILD_DIR = ../build/
EXE_DIR  = $(BUILD_DIR)EXE/
GPU_EXE  = $(BUILD_DIR)GPU_EXE/
OBJ_DIR  = $(BUILD_DIR)OBJ/
DEPS_DIR = $(BUILD_DIR)DEPS/

BUILD_SUBDIRS = $(EXE_DIR) $(GPU_EXE) $(OBJ_DIR) $(DEPS_DIR)

CXX = clang++-3.8
# CXX = g++

# put warning flags here - go to compiler *and* linker.
WARNING_FLAGS += -Wall -Wextra -pedantic -Weffc++ -Wconversion

# put other flags for both the compiler & linker here
EXTRA_FLAGS = -std=c++14

# add flags for debugging
ifeq ($(BUILD_MODE),debug)
	EXTRA_FLAGS += -ggdb -D DEBUG # -D _GLIBCXX_DEBUG
endif

# add flags for release
ifeq ($(BUILD_MODE),release)
	EXTRA_FLAGS += -O3
endif

LIBRARY_LINK_FLAGS += \
	-lstdc++ \


INCLUDE_FLAGS += \
	-I . \


CXXFLAGS += $(EXTRA_FLAGS) $(WARNING_FLAGS) $(INCLUDE_FLAGS)
LDFLAGS  += $(EXTRA_FLAGS) $(WARNING_FLAGS) $(LIBRARY_LINK_FLAGS)

GPUCC = nvcc
GPU_EXTRA_FLAGS += -ccbin clang-3.8 -Wno-deprecated-gpu-targets
GPU_LIBRARY_LINK_FLAGS += $(LIBRARY_LINK_FLAGS)
GPU_INCLUDE_FLAGS += $(INCLUDE_FLAGS)
GPUCCFLAGS += $(GPU_EXTRA_FLAGS) $(GPU_WARNING_FLAGS) $(GPU_INCLUDE_FLAGS)
GPULDFLAGS += $(GPU_EXTRA_FLAGS) $(GPU_WARNING_FLAGS) $(GPU_LIBRARY_LINK_FLAGS)

# keep .o files
.PRECIOUS: $(OBJ_DIR)%.o

# define source directories
SOURCE_DIRS = basic/ ./

ALL_OBJ_DIRS  = $(addprefix $(OBJ_DIR),  $(SOURCE_DIRS))
ALL_DEPS_DIRS = $(addprefix $(DEPS_DIR), $(SOURCE_DIRS))

# compute all directories that might need creation
DIRS= \
	$(ALL_OBJ_DIRS) \
	$(ALL_DEPS_DIRS) \
	$(BUILD_SUBDIRS) \
	$(BUILD_DIR)

# define executables
TEST_EXES=
EXES=$(EXE_DIR)simplex-cpu-only $(GPU_EXE)simplex $(TEST_EXES)

all: $(EXES) test | build_info

test: $(patsubst %, run_%, $(TEST_EXES))

build_info:
	@echo "Building with makeflags ${MAKEFLAGS}"
	@echo "In build mode ${BUILD_MODE}"

# add more dependencies here:
$(EXE_DIR)simplex-cpu-only: \
	$(OBJ_DIR)cpu_only_main.o \


$(GPU_EXE)simplex: \
	$(OBJ_DIR)gpu_cpu_main.o \


# include all the dependency files, if any exist
EXISTING_DEP_FILES = \
	$(foreach dir,$(SOURCE_DIRS), \
		$(wildcard $(DEPS_DIR)$(dir)*.d) \
	)
ifneq (,$(EXISTING_DEP_FILES))
include $(EXISTING_DEP_FILES)
endif

# compile .c++ to .o
# second CC line generates the initial dependency file
# first sed line adds $(OBJ_DIR) prefixes to the dependency file,
# second one adds stub rules for each depended on file (make might
# complain with generated files)
.SECONDEXPANSION:
$(OBJ_DIR)%.o: %.cpp | build_info $(OBJ_DIR)$$(dir %) $(DEPS_DIR)$$(dir %)
	'$(CXX)' -c  '$(shell readlink --canonicalize $<)' -o  '$@' $(CXXFLAGS)
	@'$(CXX)' -MM '$<' -MF '$(DEPS_DIR)$<.d.tmp' $(CXXFLAGS)
	@sed -e 's|.*:|$@:|' < '$(DEPS_DIR)$<.d.tmp' > '$(DEPS_DIR)$<.d'
	@sed -e 's/.*://' -e 's/\\$$//' < '$(DEPS_DIR)$<.d.tmp'| fmt -1 | \
	 sed -e 's/^ *//' -e 's/$$/:/' >> '$(DEPS_DIR)$<.d'
	@rm -f $(DEPS_DIR)$<.d.tmp

.SECONDEXPANSION:
$(OBJ_DIR)%.o: %.cu | build_info $(OBJ_DIR)$$(dir %)
	'$(GPUCC)' -c  '$(shell readlink --canonicalize $<)' -o  '$@' $(GPUCCFLAGS)

$(GPU_EXE)%: | build_info $(GPU_EXE)
	'$(GPUCC)' $(filter %.o,$^) -o '$@' $(GPULDFLAGS)

$(EXE_DIR)%: | build_info $(EXE_DIR)
	'$(CXX)' $(filter %.o,$^) -o '$@' $(LDFLAGS)

$(patsubst %, run_%, $(EXES)): $$(patsubst run_%, %, $$(@))
	@( $^ && echo "SUCCESS: $^" ) || echo "FAIL: $^"

$(DIRS):
	mkdir -p $@

clean: $(BUILD_DIR:%=delete_%_dir)

$(patsubst %, delete_%_dir, $(BUILD_DIR)    ): $$(patsubst %, delete_%_dir, $(BUILD_SUBDIRS))
$(patsubst %, delete_%_dir, $(OBJ_DIR)      ): $$(patsubst %, delete_%_dir, $$(ALL_OBJ_DIRS))
$(patsubst %, delete_%_dir, $(DEPS_DIR)     ): $$(patsubst %, delete_%_dir, $$(ALL_DEPS_DIRS))
$(patsubst %, delete_%_dir, $(EXE_DIR)      ): $$(patsubst %, delete_%_file/, $$(EXES))
$(patsubst %, delete_%_dir, $(ALL_OBJ_DIRS) ): $$(patsubst delete_%_dir, delete_%*.o_file/, $$(@))
$(patsubst %, delete_%_dir, $(ALL_DEPS_DIRS)): $$(patsubst delete_%_dir, delete_%*.d_file/, $$(@))

delete_%_file/:
	@$(eval DELFILE = $(@:delete_%_file/=%))
	@for f in $(DELFILE); do if [ $$f ]; then echo rm -f $$f; fi; done;
	@for f in $(DELFILE); do if [ $$f ]; then      rm -f $$f; fi; done;

$(DIRS:%=delete_%_dir):
	@$(eval DELDIR = $(@:delete_%_dir=%))
	@if [ -d "$(DELDIR)" ]; then echo rmdir "$(DELDIR)"; fi
	@if [ -e "$(DELDIR)" ]; then      rmdir --ignore-fail-on-non-empty "$$(readlink --canonicalize "$(DELDIR)")"; fi

endif
