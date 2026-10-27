# Post-M10 Solver Direction

The original implementation plan is paused after M10. Its M0–M10 work remains
the first technical MVP: it proved the end-to-end application workflow, but it
did not establish a sufficiently complete or extensible Railbound solver model.
M11 and M12 are therefore deferred rather than started from incomplete solver
semantics.

The next phase will initially proceed from current findings without a detailed
milestone plan. Its goal is not to solve one particular level or immediately
cover every late-game device. The goal is a dependable solver kernel built on:

- a port-level topology graph for rail placement;
- explicit deterministic transitions over a canonical runtime-state graph;
- core multi-car movement, collisions, ordered coupling, inventory, crossings,
  and placeable junctions;
- extension boundaries that let later mechanics add compiled data, state,
  transition rules, effects, events, and safe pruning without restructuring the
  solver.

Individual levels, including 1-13, are regression evidence rather than product
goals. Once the solver kernel is demonstrated, subsequent mechanics and product
features will be developed through new feature-specific plans.

Implementation through M9 inclusive used Terra at extra-high reasoning effort.
M10 used Sol at high reasoning effort. Subsequent work will use Sol at high
reasoning effort.
