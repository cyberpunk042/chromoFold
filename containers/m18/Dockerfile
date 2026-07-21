FROM debian:13-slim@sha256:9d67a0a4a2d857f7253f11d5c96b9f84c7381a03f8afbdc574bf53aee8f5a764 AS runtime

RUN groupadd --gid 65532 chromofold \
    && useradd --uid 65532 --gid 65532 --home-dir /nonexistent --shell /usr/sbin/nologin chromofold \
    && mkdir -p /opt/chromofold /var/lib/chromofold /run/chromofold \
    && chown -R 65532:65532 /opt/chromofold /var/lib/chromofold /run/chromofold \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=65532:65532 build/release/ /opt/chromofold/

USER 65532:65532
WORKDIR /opt/chromofold
ENV HOME=/nonexistent \
    CHROMOFOLD_STATE_DIR=/var/lib/chromofold \
    CHROMOFOLD_RUN_DIR=/run/chromofold
VOLUME ["/var/lib/chromofold"]
EXPOSE 8080
ENTRYPOINT ["/opt/chromofold/chromofold-server"]
