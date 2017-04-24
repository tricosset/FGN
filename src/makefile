CXXFLAGS :=-O3 -std=c++11 -fPIC -I./fgn -I.
TAR= demo/fgn_dr_mp_demo
OBJ= fgn/fastrandombytes.o fgn/randombytes.o fgn/crypto_stream_salsa20_amd64_xmm6.o

all: $(TAR)

crypto_stream_salsa20_amd64_xmm6.o: crypto_stream_salsa20_amd64_xmm6.s
	$(CXX) -c $(CXXFLAGS) $?

%.o: %.cpp
	$(CXX) -c $< $(CXXFLAGS) -o $@

$(TAR): % : %.o fgn/libfastgaussian.so
	$(CXX) $(CXXFLAGS) $^ -lmpfr -lgmp -o $@

fgn/libfastgaussian.so: $(OBJ) fgn.hpp
	$(CXX) $(CXXFLAGS) $^ -shared -lmpfr -lgmp -o $@
	cp fgn/libfastgaussian.so libfastgaussian.so
	#cp fgn/libfastgaussian.so /usr/lib/libfastgaussian.so

clean:
	$(RM) $(TAR) demo/*.o fgn/*.o fgn/*.gch fgn/libfastgaussian.so libfastgaussian.so
