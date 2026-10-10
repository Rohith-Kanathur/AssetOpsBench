ARG ZCODE_REV=29628c9acdb81b703bbd4080c207a0e7ce5e276e
FROM node:24.14.0-bookworm AS build
ARG ZCODE_REV
ARG TARGETARCH
ENV ELECTRON_SKIP_BINARY_DOWNLOAD=1 ZCODE_ENV=production CI=1
WORKDIR /opt/zcode
RUN git init && git remote add origin https://github.com/zai-org/ZCode.git \
    && git fetch --depth=1 origin "$ZCODE_REV" && git checkout --detach FETCH_HEAD
RUN npm install -g pnpm@10.33.2
RUN pnpm --filter @zcode/cli... install --frozen-lockfile
RUN pnpm --workspace-concurrency=1 --filter @zcode/cli... build
RUN rm -f apps/zcode-cli/packages/cli/dist/zcode.cjs.map \
    && mkdir /opt/runtime-tools \
    && case "$TARGETARCH" in arm64) zcode_arch=aarch64;; amd64) zcode_arch=x86_64;; *) exit 1;; esac \
    && cd apps/zcode-cli/dependencies/native-search && sha256sum --check --ignore-missing SHA256SUMS \
    && for archive in bfs-v4.1.1-2/*-"$zcode_arch"-unknown-linux-gnu.tar.gz \
        ugrep-v7.8.4-1/*-"$zcode_arch"-unknown-linux-gnu.tar.gz \
        ripgrep-v14.1.1-1/*-"$zcode_arch"-unknown-linux-musl.tar.gz; do \
        tar -xzf "$archive" -C /opt/runtime-tools; done

FROM assetops-scenario-evaluation:local
ARG ZCODE_REV
COPY --from=build /usr/local/bin/node /opt/zcode-node/bin/node
COPY --from=build /opt/zcode/apps/zcode-cli/packages/cli/dist /opt/zcode/apps/zcode-cli/packages/cli/dist
COPY --from=build /opt/zcode/config/provider/zcode-builtin.json /opt/zcode/config/provider/zcode-builtin.json
COPY --from=build /opt/runtime-tools /opt/zcode/runtime-tools
LABEL org.assetops.zcode.source_commit=$ZCODE_REV
RUN /opt/zcode-node/bin/node /opt/zcode/apps/zcode-cli/packages/cli/dist/zcode.cjs --help
