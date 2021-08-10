# Stage 0 - Create from hivdi:0.3 image
FROM hivdi:0.4b as stage0

# Stage 1 - Copy script for running algorithm (AWS pre-benchmark version)
FROM stage0 as stage1
COPY run_hivdi.py /app/run_hivdi.py

# Stage 2 - Execute algorithm
FROM stage1 as stage2
LABEL version="1.0" \
	description="AWS HiVDI algorithm." \
	"confluence.contact"="ntebaldi@umass.edu" \
	"algorithm.contact"="kevin.larnier@csgroup.eu"
# Environment variable to enforce 20 iterations VDA+virtuals
ENV VRT_ITERATIONS_COUNT=50
ENTRYPOINT ["/usr/bin/python3", "/app/run_hivdi.py"]