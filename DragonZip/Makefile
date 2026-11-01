# DragonZip — yig'ish
#   make            oddiy (har qanday x86-64 / ARM kompyuterda ishlaydi)
#   make native     shu kompyuter protsessoriga moslab (AVX2 bo'lsa tezroq)
#   make test       avtomatik sinovlar
CXX      ?= g++
CXXFLAGS ?= -O2
FLAGS     = -std=c++17 -pthread -Wall -Wextra
TARGET    = dragonzip

all: $(TARGET)

$(TARGET): dragonzip.cpp
	$(CXX) $(CXXFLAGS) $(FLAGS) dragonzip.cpp -o $(TARGET)

native: dragonzip.cpp
	$(CXX) -O2 -march=native $(FLAGS) dragonzip.cpp -o $(TARGET)

test: $(TARGET)
	./tests/run_tests.sh ./$(TARGET)

install: $(TARGET)
	install -m 755 $(TARGET) /usr/local/bin/$(TARGET)

clean:
	rm -f $(TARGET) $(TARGET).exe

.PHONY: all native test install clean
