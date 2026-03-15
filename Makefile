# Makefile for adaptive-edge-heuristic

CXX      := g++
CXXFLAGS := -O3 -std=c++11 -Wall -Wextra -fPIC -Icpp
SRCS     := cpp/engine.cpp cpp/simulator.cpp cpp/learner.cpp

.PHONY: all clean run

# ── Linux (default) ───────────────────────────────────────────────────────────
all: cpp/libheuristic.so

cpp/libheuristic.so: $(SRCS)
	$(CXX) $(CXXFLAGS) -shared -o $@ $^
	@echo "Built $@"

# ── macOS ─────────────────────────────────────────────────────────────────────
macos: $(SRCS)
	$(CXX) $(CXXFLAGS) -dynamiclib -o cpp/libheuristic.dylib $^
	@echo "Built cpp/libheuristic.dylib"

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -f cpp/libheuristic.so cpp/libheuristic.dylib cpp/heuristic.dll

# ── Run headless driver ───────────────────────────────────────────────────────
run: cpp/libheuristic.so
	python python/headless_driver.py
