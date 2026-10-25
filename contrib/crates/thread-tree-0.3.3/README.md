## Thread tree

A tree-structured thread pool. See [API documentation](https://docs.rs/thread-tree) for more information.

Stack jobs and job execution based on rayon-core by Niko Matsakis and Josh Stone.

Experimental simple thread pool used for spawning stack-bound scoped jobs with no work stealing.

This is good for:

- You want to split work recursively in jobs that use approximately the same time.
- You want thread pool overhead to be low

This is not good for:

- You need work stealing
- When you have jobs of uneven size


### Wild ideas and notes

Possibly allow reserving a subbranch of the tree.

