FROM ubuntu:22.04

# ---------- Proxy settings for git only (passed in at build time) ----------
ARG GIT_HTTP_PROXY
ARG GIT_HTTPS_PROXY

# ------------------------------------------------------------
# Install build tooling and dependencies (unset all proxies explicitly here)
# ------------------------------------------------------------
RUN unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential \
        cmake \
        clang \
        ninja-build \
        curl \
        python3 \
        git \
        zip \
        unzip \
        tar \
    	libabsl-dev \
        libssl-dev && \
    apt-get clean

# ------------------------------------------------------------
# Install vcpkg (Microsoft C/C++ package manager) — proxy applies to git only
# ------------------------------------------------------------
RUN git -c http.proxy="${GIT_HTTP_PROXY}" \
        -c https.proxy="${GIT_HTTPS_PROXY}" \
    clone https://github.com/microsoft/vcpkg.git /opt/vcpkg && \
    /opt/vcpkg/bootstrap-vcpkg.sh

# ------------------------------------------------------------
# Working directory
# ------------------------------------------------------------
WORKDIR /app

# ------------------------------------------------------------
# 修复 Git 安全性检查
# ------------------------------------------------------------
RUN git config --global --add safe.directory /app && \
    git config --global --add safe.directory /app/duckdb

# ------------------------------------------------------------
# 默认构建类型
# ------------------------------------------------------------
ENV BUILD_TYPE=release

#CMD ["bash", "-c", "\
#  export CMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake && \
#  export OVERRIDE_GIT_DESCRIBE='v0.10.3' && \
#  export GIT_COMMIT_HASH='dummy' && \
#  case \"${BUILD_TYPE}\" in \
#    debug) \
#      export EXT_DEBUG_FLAGS='-DCMAKE_CXX_FLAGS_DEBUG=\"-g3 -O0 -fno-omit-frame-pointer -fno-inline -gdwarf-4\" -DCMAKE_C_FLAGS_DEBUG=\"-g3 -O0 -fno-omit-frame-pointer -gdwarf-4\" -DCMAKE_SHARED_LINKER_FLAGS_DEBUG=\"-Wl,--build-id\"' && \
#      make debug \
#      ;; \
#    reldebug) \
#      export EXT_RELEASE_FLAGS='-DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS_RELWITHDEBINFO=\"-g -O2 -fno-omit-frame-pointer\" -DCMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO=\"-Wl,--build-id\"' && \
#      make reldebug \
#      ;; \
#    *) \
#      make release \
#      ;; \
#  esac \
#"]
CMD ["bash", "-lc", "\
  export OVERRIDE_GIT_DESCRIBE='v0.10.3' && \
  export GIT_COMMIT_HASH='dummy' && \
  case \"${BUILD_TYPE}\" in \
    debug) \
      export EXT_DEBUG_FLAGS='\
        -DCMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake \
        -DVCPKG_MANIFEST_MODE=ON \
        -DVCPKG_MANIFEST_INSTALL=ON \
        -DVCPKG_MANIFEST_DIR=/app \
        -DCMAKE_CXX_FLAGS_DEBUG=\"-g3 -O0 -fno-omit-frame-pointer -fno-inline -gdwarf-4\" \
        -DCMAKE_C_FLAGS_DEBUG=\"-g3 -O0 -fno-omit-frame-pointer -gdwarf-4\" \
        -DCMAKE_SHARED_LINKER_FLAGS_DEBUG=\"-Wl,--build-id\"' && \
      make debug \
      ;; \
    reldebug) \
      export EXT_RELEASE_FLAGS='\
        -DCMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake \
        -DVCPKG_MANIFEST_MODE=ON \
        -DVCPKG_MANIFEST_INSTALL=ON \
        -DVCPKG_MANIFEST_DIR=/app \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_CXX_FLAGS_RELWITHDEBINFO=\"-g -O2 -fno-omit-frame-pointer\" \
        -DCMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO=\"-Wl,--build-id\"' && \
      make reldebug \
      ;; \
    *) \
      export EXT_RELEASE_FLAGS='\
        -DCMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake \
        -DVCPKG_MANIFEST_MODE=ON \
        -DVCPKG_MANIFEST_INSTALL=ON \
        -DVCPKG_MANIFEST_DIR=/app' && \
      make release \
      ;; \
  esac \
"]

