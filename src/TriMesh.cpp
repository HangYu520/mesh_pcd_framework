#include "TriMesh.h"
#include <unordered_map>
#include <igl/per_vertex_normals.h>
#include <igl/per_face_normals.h>
#include <igl/massmatrix.h>
#include <igl/cotmatrix.h>
#include <igl/invert_diag.h>
#include <igl/principal_curvature.h>
#include <igl/gaussian_curvature.h>
#include <igl/avg_edge_length.h>
#include <igl/barycenter.h>
#include <igl/grad.h>
#include <igl/doublearea.h>
#include <igl/loop.h>
#include <igl/writeOBJ.h>
#include <igl/marching_cubes.h>

void TriMesh::ReadFromFile(const char* filepath)
{
	igl::read_triangle_mesh(filepath, m_vertices, m_faces);
}

void TriMesh::writeOBJ(const char* filepath)
{
	if (Empty())
		return;
	igl::writeOBJ(filepath, m_vertices, m_faces);
}

void TriMesh::SetMesh(Mesh& mesh)
{
	int n_vertices = mesh.number_of_vertices();
	int n_faces = mesh.number_of_faces();
	Eigen::MatrixXd V(n_vertices, 3);
	Eigen::MatrixXi F(n_faces, 3);
	//set vertex
	int i = 0;
	for (vertex_descriptor vd : mesh.vertices())
	{
		Point_3 vertex = mesh.point(vd);
		V(i, 0) = vertex.x();
		V(i, 1) = vertex.y();
		V(i, 2) = vertex.z();
		i++;
	}
	i = 0;
	//set face
	for (face_descriptor fd : mesh.faces())
	{
		halfedge_descriptor hd = mesh.halfedge(fd);
		int j = 0;
		for (vertex_descriptor vd : mesh.vertices_around_face(hd))
		{
			F(i, j) = vd.idx();
			j++;
		}
		i++;
	}

	m_vertices = V;
	m_faces = F;
}

Mesh TriMesh::getMesh()
{
	Mesh m;
	std::unordered_map<int, vertex_descriptor> vd_map;
	//add vertex
	for (int i = 0; i < n_vertices(); i++)
	{
		Point_3 p(m_vertices(i, 0), m_vertices(i, 1), m_vertices(i, 2));
		vertex_descriptor v = m.add_vertex(p);
		vd_map[i] = v;
	}
	//add face
	for (int j = 0; j < n_faces(); j++)
	{
		vertex_descriptor u = vd_map[m_faces(j, 0)];
		vertex_descriptor v = vd_map[m_faces(j, 1)];
		vertex_descriptor w = vd_map[m_faces(j, 2)];
		m.add_face(u, v, w);
	}
	return m;
}

std::unordered_map<int, Color> TriMesh::ColorMap(int num)
{
	//return a colormap contains num colors
	std::unordered_map<int, Color> colormap;
	boost::mt19937 zgy;  //Uniform pseudorandom number generator
	zgy.seed(static_cast<unsigned int>(time(0)));	//random seed
	boost::uniform_real<> ur(0, 1);
	for (int i = 0; i < num; i++)
	{
		colormap[i] = Color{ ur(zgy), ur(zgy), ur(zgy) };
	}
	return colormap;
}

void TriMesh::Draw(igl::opengl::glfw::Viewer& viewer) const
{
	if (Empty())
		return;
	viewer.data().clear();
	viewer.data().set_mesh(m_vertices, m_faces);
	viewer.core().align_camera_center(m_vertices, m_faces);
}

void TriMesh::DrawGaussCurvature(igl::opengl::glfw::Viewer& viewer)
{
	auto K = gaussCurvature();
	auto Minv = massMatrixInv();
	K = (Minv * K).eval();
	viewer.data().set_data(K);
}

void TriMesh::DrawMeanCurvature(igl::opengl::glfw::Viewer& viewer)
{
	auto H = meanCurvature();
	viewer.data().set_data(H);
}

void TriMesh::DrawSegmentation(igl::opengl::glfw::Viewer& viewer, std::vector<int>& faceseg)
{
	//faceseg is a n_faces dimensional vector (the type of class)
	std::unordered_set<int> type_set;
	for (int fg : faceseg)
		type_set.insert(fg);
	int n_types = type_set.size();
	std::unordered_map<int, Color> colormap = ColorMap(n_types);
	Eigen::MatrixXd C(n_faces(), 3);
	for (int i = 0; i < n_faces(); i++)
	{
		C(i, 0) = colormap[faceseg[i]].at(0);
		C(i, 1) = colormap[faceseg[i]].at(1);
		C(i, 2) = colormap[faceseg[i]].at(2);
	}
	viewer.data().set_colors(C);
}

void TriMesh::DrawData(igl::opengl::glfw::Viewer& viewer, std::vector<double>& nodedata)
{
	// nodedata is a n_vertices*1 vector
	Eigen::VectorXd D(n_vertices(), 1);
	for (int i = 0; i < n_vertices(); i++)
	{
		D(i) = nodedata[i];
	}
	viewer.data().set_data(D);
}

void TriMesh::Drawboundary(igl::opengl::glfw::Viewer& viewer)
{
	Eigen::MatrixXd boundary_points = boundary();
	Eigen::MatrixXd C = boundary_points;
	for (int i = 0; i < C.rows(); i++)
		C.row(i) = Eigen::RowVector3d(1, 0, 0);
	viewer.data().add_points(boundary_points, C);
}

double TriMesh::AverageEdgeLength()
{
	return igl::avg_edge_length(m_vertices, m_faces);
}

Eigen::MatrixXd TriMesh::perVertexNormal()
{
	//return normals of each vertex
	Eigen::MatrixXd N_vertices;
	igl::per_vertex_normals(m_vertices, m_faces, N_vertices);
	return N_vertices;
}

Eigen::MatrixXd TriMesh::perFaceNormal()
{
	//return normals of each face
	Eigen::MatrixXd N_faces;
	igl::per_face_normals(m_vertices, m_faces, N_faces);
	return N_faces;
}

Eigen::SparseMatrix<double> TriMesh::massMatrix(igl::MassMatrixType type)
{
	//return mass matrix M
	Eigen::SparseMatrix<double> M;
	igl::massmatrix(m_vertices, m_faces, type, M);
	return M;
}

Eigen::SparseMatrix<double> TriMesh::massMatrixInv(igl::MassMatrixType type)
{
	//return invert mass matrix Inv(M)
	Eigen::SparseMatrix<double> M = massMatrix(type);
	Eigen::SparseMatrix<double> Minv;
	igl::invert_diag(M, Minv);
	return Minv;
}

Eigen::SparseMatrix<double> TriMesh::cotLaplace()
{
	//return Laplace-Beltrami operator
	Eigen::SparseMatrix<double> L;
	igl::cotmatrix(m_vertices, m_faces, L);
	return L;
}

TriMesh::PrincipalCurvature TriMesh::principalCurvature()
{
	PrincipalCurvature p;
	igl::principal_curvature(m_vertices, m_faces, p.PD1, p.PD2, p.PV1, p.PV2);
	return p;
}

Eigen::VectorXd TriMesh::gaussCurvature()
{
	//return gauss curvature
	Eigen::VectorXd K;
	igl::gaussian_curvature(m_vertices, m_faces, K);
	return K;
}

Eigen::VectorXd TriMesh::meanCurvature()
{
	//return mean curvature
	Eigen::VectorXd H;
	auto p = principalCurvature();
	return 0.5 * (p.PV1 + p.PV2);
}

Eigen::MatrixXd TriMesh::baryCenter()
{
	//return barycenter of each face
	Eigen::MatrixXd BC;
	igl::barycenter(m_vertices, m_faces, BC);
	return BC;
}

Eigen::MatrixXd TriMesh::Grad(Eigen::VectorXd& value)
{
	//return the face gradient of value(defined on vertex)
	Eigen::SparseMatrix<double> G;
	igl::grad(m_vertices, m_faces, G);
	Eigen::MatrixXd GU = Eigen::Map<const Eigen::MatrixXd>((G * value).eval().data(), n_faces(), 3);
	return GU;
}

double TriMesh::faceArea(int face_id)
{
	Eigen::MatrixXd vertices = Vertices();
	Eigen::MatrixXi faces = Faces();
	int u = faces(face_id, 0);
	int v = faces(face_id, 1);
	int w = faces(face_id, 2);
	Point_3 p(vertices(u, 0), vertices(u, 1), vertices(u, 2));
	Point_3 q(vertices(v, 0), vertices(v, 1), vertices(v, 2));
	Point_3 r(vertices(w, 0), vertices(w, 1), vertices(w, 2));
	return sqrt(CGAL::squared_area(p, q, r));
}

Eigen::MatrixXd TriMesh::boundary()
{
	std::vector<std::vector<int>> boundaries;
	std::vector<int> boundary;
	igl::boundary_loop(m_faces, boundaries);
	for (int i = 0; i < boundaries.size(); i++)
		for (int j = 0; j < boundaries[i].size(); j++)
			boundary.push_back(boundaries[i][j]);
	Eigen::MatrixXd boundary_points(boundary.size(), 3);
	for (int i = 0; i < boundary.size(); i++)
			boundary_points.row(i) = m_vertices.row(boundary[i]);
	return boundary_points;
}

void TriMesh::LaplaceSmooth()
{
	std::cout << "Laplace Smooth" << std::endl;
	//smooth once time, call function multiple times to smooth ieratively
	auto M = massMatrix(igl::MASSMATRIX_TYPE_BARYCENTRIC);
	auto L = cotLaplace();
	//solve (M-delta*L) V = M*V
	const auto& S = (M - 0.001 * L);
	Eigen::SimplicialLLT<Eigen::SparseMatrix<double > > solver(S);
	assert(solver.info() == Eigen::Success);
	m_vertices = solver.solve(M * m_vertices).eval();
	// Compute centroid and subtract (also important for numerics)
	Eigen::VectorXd dblA;
	igl::doublearea(m_vertices, m_faces, dblA);
	double area = 0.5 * dblA.sum();
	Eigen::MatrixXd BC = baryCenter();
	Eigen::RowVector3d centroid(0, 0, 0);
	for (int i = 0; i < BC.rows(); i++)
	{
		centroid += 0.5 * dblA(i) / area * BC.row(i);
	}
	m_vertices.rowwise() -= centroid;
	// Normalize to unit surface area (important for numerics)
	m_vertices.array() /= sqrt(area);
}

void TriMesh::LoopSubdiv(int number_of_subdivs)
{
	//loop subdivision
	Eigen::MatrixXd NV;
	Eigen::MatrixXi NF;
	igl::loop(m_vertices, m_faces, NV, NF, number_of_subdivs);
	m_vertices = NV;
	m_faces = NF;
}

void TriMesh::MarchingCubes(Eigen::VectorXd& S, Eigen::MatrixXd& GV,
	int nx, int ny, int nz, double isovalue)
{
	//marching cubes algorithm to extract isovalue face
	/*
	* S: nx*ny*nz list of values at each grid corner
  //       i.e. S(x + y*xres + z*xres*yres) for corner (x,y,z)
	* GV: nx*ny*nz by 3 array of corresponding grid corner vertex locations
	* nx  resolutions of the grid in x dimension
	* ny  resolutions of the grid in y dimension
    * nz  resolutions of the grid in z dimension
	* isovalue: the isovalue of the surface to reconstruct
	*/
	std::cout << "Reconstruct iso-face via marching cubes" << std::endl;
	igl::marching_cubes(S, GV, nx, ny, nz, isovalue, m_vertices, m_faces);
	std::cout << "Done" << std::endl;
}