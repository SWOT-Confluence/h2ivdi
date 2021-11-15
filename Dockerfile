# Stage 0 - Create from hivdi:0.5c image
FROM h2ivdi:0.5c as stage0

# Stage 1 - Copy script for running algorithm (AWS pre-benchmark version)
FROM stage0 as stage1
COPY run_h2ivdi.py /app/run_h2ivdi.py

# Stage 2 - Execute algorithm
FROM stage1 as stage2
LABEL version="0.5c" \
	description="AWS H2iVDI algorithm." \
	"confluence.contact"="ntebaldi@umass.edu" \
	"algorithm.contact"="kevin.larnier@csgroup.eu"
# Environment variable to enforce 20 iterations VDA+virtuals
ENV VRT_ITERATIONS_COUNT 20
# Entry point
ENTRYPOINT  ["/usr/bin/python3", "/app/run_h2ivdi.py" ]