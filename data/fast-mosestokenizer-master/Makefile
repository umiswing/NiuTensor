.PHONY = help build install clean

help:
	@echo "Subcommands"
	@echo "==========="
	@echo "- build"
	@echo "	Build the mosestokenizer c library"
	@echo "- clean"
	@echo "	Remove all build files"
	@echo "- download-build-static-deps"
	@echo "	Download and build static dependencies"

build:
	mkdir -p build/rel
	( \
		cd build/rel; \
		cmake ../.. \
			-DCMAKE_BUILD_TYPE=Release \
			-DBUILD_SHARED_LIBS:BOOL=ON \
			-DBUILD_CLI:BOOL=ON; \
		cmake --build . -j --config Release; \
	)

install:
	cd build/rel && make install

clean:
	rm -rf build bindings/python/mosestokenizer/lib

download-build-static-deps:
	@mkdir -p deps

	@echo "Downloading and building re2"
	curl -L -o deps/re2-2020-06-01.tar.gz \
		https://github.com/google/re2/archive/2020-06-01.tar.gz
	tar -C deps -xf deps/re2-2020-06-01.tar.gz
	cd deps/re2-2020-06-01; CXXFLAGS="-fPIC" make
