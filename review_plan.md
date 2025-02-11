# Code Review and Improvement Plan

## Current Issues

1. **NamedTuple Usage**
   - BaseEstimate and SubsampleIndices are implemented as NamedTuples
   - This limits extensibility and makes the code less maintainable
   - No easy way to add methods or modify behavior

2. **Code Organization**
   - Enums (LooApproximationMethod, EstimatorMethod) are in loo_subsample.py
   - APPROXIMATION_METHODS dictionary creates tight coupling
   - Type hints could be more elegant

3. **Module Structure**
   - Estimators package could be better organized
   - No clear separation between interfaces and implementations

## Proposed Improvements

1. **Convert to @dataclass**
   ```python
   from dataclasses import dataclass

   @dataclass
   class BaseEstimate:
       y_hat: float
       v_y_hat: float
       hat_v_y: float
       m: int
       subsampling_SE: float
       N: int = 0
   ```

2. **Create Constants Module**
   - Move Enums to pyloo/pyloo/constants.py
   - Use string literals for better type safety
   - Add proper documentation

3. **Implement Registry Pattern**
   ```python
   from typing import Protocol, Dict, Type

   class ApproximationMethod(Protocol):
       def compute_approximation(self, log_likelihood, n_draws=None): ...

   class ApproximationRegistry:
       _methods: Dict[str, Type[ApproximationMethod]] = {}

       @classmethod
       def register(cls, name: str):
           def decorator(method_cls: Type[ApproximationMethod]):
               cls._methods[name] = method_cls
               return method_cls
           return decorator
   ```

4. **Factory Pattern for Estimators**
   ```python
   class EstimatorFactory:
       @staticmethod
       def create(method: str, **kwargs) -> BaseEstimate:
           if method == "diff_srs":
               return diff_srs_estimate(**kwargs)
           elif method == "hh_pps":
               return hansen_hurwitz_estimate(**kwargs)
           elif method == "srs":
               return srs_estimate(**kwargs)
           raise ValueError(f"Unknown estimator method: {method}")
   ```

5. **Improved Type Hints**
   - Use Protocol for better interface definitions
   - Add proper type hints for all functions
   - Use TypeVar for generic types where appropriate

## Implementation Steps

1. Create new constants.py module
2. Convert BaseEstimate to @dataclass
3. Implement ApproximationRegistry
4. Create EstimatorFactory
5. Update all estimator implementations
6. Update loo_subsample.py to use new patterns
7. Update tests to reflect changes

## Benefits

1. **Better Maintainability**
   - Clear separation of concerns
   - More organized code structure
   - Easier to add new methods

2. **Improved Type Safety**
   - Better type hints
   - Protocol-based interfaces
   - Clearer error messages

3. **Enhanced Extensibility**
   - Easy to add new approximation methods
   - Simple to extend estimators
   - Clear patterns for future development

4. **Cleaner Code**
   - Less coupling between modules
   - More intuitive organization
   - Better documentation

## Testing Strategy

1. Ensure all existing tests pass with new implementation
2. Add tests for new registry and factory patterns
3. Verify type hints work correctly
4. Test error cases and edge conditions

## Migration Plan

1. Create new implementations alongside existing code
2. Gradually migrate functionality
3. Update tests in parallel
4. Remove old implementations once verified

## Notes

- All changes maintain backward compatibility
- No changes to core algorithms or mathematical implementations
- Focus on code organization and maintainability
- Preserve existing test coverage

Would you like me to proceed with implementing these changes?
