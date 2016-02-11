initSidebarItems({"enum":[["IndexType","This enum serves to represent the type of a tensor. A tensor can have any number of indices, and each one can be either covariant (a lower index), or contravariant (an upper index). For example, a vector is a tensor with only one contravariant index."]],"struct":[["ContravariantIndex","Type representing a contravariant (upper) tensor index."],["CovariantIndex","Type representing a covariant (lower) tensor index."],["Tensor","Struct representing a tensor."]],"trait":[["Concat","Operator trait used for concatenating two variances."],["Contract","An operator trait representing tensor contraction"],["InnerProduct","Trait representing the inner product of two tensors."],["OtherIndex","Trait representing the other index type"],["TensorIndex","Trait identifying a type as representing a tensor index. It is implemented for `CovariantIndex` and `ContravariantIndex`."],["Variance","Trait identifying a type as representing a tensor variance. It is implemented for `CovariantIndex`, `ContravariantIndex` and tuples (Index, Variance)."]],"type":[["Contracted","Helper type for contraction"],["Covector","A covector type (rank 1 covariant tensor)"],["InvTwoForm","A rank 2 doubly contravariant tensor"],["Joined","Helper type for variance concatenation."],["Matrix","A matrix type (rank 2 contravariant-covariant tensor)"],["TwoForm","A bilinear form type (rank 2 doubly covariant tensor)"],["Vector","A vector type (rank 1 contravariant tensor)"]]});