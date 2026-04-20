CXX      := c++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -pedantic
TARGET   := problème/engine
SOURCE   := problème/engine.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCE) common/ConfigFile.h common/ConfigFile.hpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE)

clean:
	rm -f $(TARGET) problème/output*.out
