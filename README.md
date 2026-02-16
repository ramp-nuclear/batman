# Burnup Code
This code was designed to solve the Bateman's equation:
dN/dt = (Lambda+React*Phi)*N
where N is the vector of nuclide density for each nuclide in each spatial cell,
Lambda is the decay model matrix, that corresponds to the rate of decay of each
nuclide to other nuclides, React is the reaction model matrix, that corresponds
to the rate of transmutation of each isotope into other isotopes due to the
neutron flux, and Phi is the absolute flux level in the core that has to be
somehow normalized to make the reaction rates fit the actual physical processes.

## Design Goals
The design goals for this package are:
1. The code must be parallelizable to more than one computer (achieved).
1. The code must pass benchmarks (partially achieved).
1. Code runtime must be in the few minute scale. CPU time may be up to an hour 
   for a large core. (Achieved! Full core at 7 minutes with 32 CPUs)
1. Code must be stable for large time steps. (Partially Achieved, depends on 
   power renormalization requirements.)
1. Code must be abstract and use simple external API. (Under Review)
1. Isotopic data must allow for both fudged and real database built data. (Achieved!)
1. Code must pass high standards (Under Review and Not Finished):
    a. Code must be fully documented. Type hinting counts for simple functions.
    b. Code must be modular and modules should be as implementation independent 
       of one another as possible.
    c. Code must pass code review regularly and Major versions in flying colors.
    d. Test coverage must be 90%+, and tests must include parallel tasked cases. 
       Mock tests are required. Multiple input tests required for numerical tests.
    e. Code must be easily installable.

## Ways to Contribute
1. We need to work on proper automatic documentation. Currently not done.
1. Code needs better quality and refactoring. You're welcome to try your hand.
1. Other exponentiation methods than CRAM could potentially be faster. Please
   try your hand at them.
1. The code currently takes explicit time steps with constant flux as the base
   method. The academic papers speak of using a predictor-corrector model to
   extend the correctness with lower cost. This should be investigated to reduce
   the number of internal steps, or a Lie algebra method can be used as well.
1. Benchmarks for this code are required.
