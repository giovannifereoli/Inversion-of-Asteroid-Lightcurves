# Shape reconstruction (Appendix C)

Inversion recovers the *areas* of the facets, not the body. Turning one into the
other is the Minkowski problem:

> The reconstruction of the convex polyhedron corresponding to given facet areas
> $g$ and surface normals can be expressed as a constrained minimization problem
> where $l$, the distances of the facet planes from the origin, are to be solved
> for. ... In practice, the equivalent procedure of maximizing $V(l)$ while staying
> on the hyperplane $\langle l, g\rangle = \text{constant}$ is computationally more
> efficient as the constraint function is now linear.

The volume is Eq. (17), its gradient is $A$, and Eq. (18) projects that gradient
onto the constraint plane:

$$V = \frac{1}{3}\sum_j l_j A_j(l), \qquad
\mathbf{f} = \mathbf{A} - \frac{\langle \mathbf{A},\mathbf{g}\rangle}{\langle \mathbf{g},\mathbf{g}\rangle}\,\mathbf{g}$$

```python
result = lcinv.minkowski_solve(geometry.normals, areas)
body = result.polyhedron
print(result.alignment)     # <A,g>/(|A||g|), 1 at the optimum
```

## How $A_j(l)$ is computed

Through the dual transform of Eq. (19), $\mathbf{r} = \mathbf{n}/l$:

> The important point is that adjacency information is retained; i.e., the vertices
> of a facet become the facets surrounding a vertex.

Because the transform is an involution, the hull of the dual points has one facet
per *vertex* of the home body. Dualising those facets back gives the home
vertices, and the body is their convex hull. Each of its triangles is then
attributed to the requested normal it is parallel to, which recovers $A_j$.

## Two details worth knowing

**The final scaling.** Appendix C ends by scaling each vertex coordinate by
$\sqrt{|A|/|g|}$. The constraint fixes $\langle l,g\rangle$ but not the scale of the
*areas*, so at the optimum $\mathbf{A} = c\,\mathbf{g}$ and one needs $cs^2 = 1$,
i.e. $s = \sqrt{|g|/|A|}$ — the reciprocal of the printed expression. `lcinv` uses
the equivalent, better-conditioned $s = \sqrt{\langle g,g\rangle/\langle A,g\rangle}$,
which reproduces the unit sphere exactly from its own facet areas.

**The dark facet.** A solved $g$ will not satisfy Eq. (3) exactly. Section 3.3:

> A small nonzero residual in (3) is easily fixed by adding a facet of
> corresponding size such that the new (3) vanishes ... The small new facet is
> completely dark, but its existence does not affect the overall shape.

`close_facet_areas` does this, and `MinkowskiResult.dark_area` reports how big it
had to be. The reference C implementation prints the same quantity.

Recentring each iteration ("it is useful to shift the centroid of the polyhedron to
the origin at each iteration step") is a pure gauge move: it leaves the body
unchanged, and leaves $\langle l,g\rangle$ unchanged too whenever Eq. (3) holds.

## Checks

`minkowski_solve` recovers, from their own facet areas: a cube, a $3\times2\times1$
brick (extents exact to $10^{-3}$), a sphere, and a triaxial ellipsoid. Duplicate
normals — which a triangulated hull always has on its large planar faces — are
merged before the dual transform, since identical normals map to identical dual
points.
