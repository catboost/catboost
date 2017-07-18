#pragma once

class TBuffer;
class TInputStream;

/*
 * fast entropy pool, based on good prng, can lock for some time
 * initialized with some bits from system entropy pool
 * think as /dev/urandom replacement
 */
TInputStream& EntropyPool();

/*
 * fast(non-blocking) entropy pool, useful for seeding PRNGs
 */
TInputStream& Seed();

/*
 * initial host entropy data
 */
const TBuffer& HostEntropy();
