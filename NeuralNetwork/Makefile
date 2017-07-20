CXX      = g++   # for Linux RedHat 6.1, g++ version 2.95.2

CPPFLAGS = -I	eigen-eigen-5a0156e40feb    -O  
LDFLAGS  = -L/gstream/ -lgzstream -lz
AR       = ar cr

# ----------------------------------------------------------------------------
# plain simple rules to make and cleanup the library:
# make default;   compiles the library
# make test;      compiles and executes test. O.K. message marks success.
# make clean;     removes temporary files
# make cleanall;  removes temporary files, the library, and programs
# ----------------------------------------------------------------------------


Source.o : Source.cpp
	${CXX} ${CPPFLAGS} $? -o $@ 


clean:
	rm *o 
