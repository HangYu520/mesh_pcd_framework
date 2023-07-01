#pragma once
#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/boundary_loop.h>
#include <unordered_map>
#include "Type.h"

class TriMesh
{
public:
	struct PrincipalCurvature
	{
		Eigen::MatrixXd PD1, PD2; //two principal curvature directions
		Eigen::VectorXd PV1, PV2; //two principal curvature values
	};
private:
	Eigen::MatrixXd m_vertices;  //libigl type mesh
	Eigen::MatrixXi m_faces;
public:
	~TriMesh() = default;
	TriMesh() = default;
	inline Eigen::MatrixXd Vertices() const { return m_vertices; }
	inline Eigen::MatrixXi Faces() const { return m_faces; }
	inline int n_vertices() const { return m_vertices.rows(); }
	inline int n_faces() const { return m_faces.rows(); }
	inline bool Empty() const { return n_vertices() == 0; }

	void ReadFromFile(const char* filepath);
	void writeOBJ(const char* filepath);
	void SetMesh(Mesh& mesh); //set a cgal mesh to igl mesh
	Mesh getMesh(); //get a cgal mesh from igl mesh
	std::unordered_map<int, Color> ColorMap(int num); //generate a colormap
	void Draw(igl::opengl::glfw::Viewer& viewer) const;
	void DrawGaussCurvature(igl::opengl::glfw::Viewer& viewer);
	void DrawMeanCurvature(igl::opengl::glfw::Viewer& viewer);
	void DrawSegmentation(igl::opengl::glfw::Viewer& viewer, std::vector<int>& faceseg);
	void DrawData(igl::opengl::glfw::Viewer& viewer, std::vector<double>& nodedata);
	void Drawboundary(igl::opengl::glfw::Viewer& viewer);
	//operation
	double AverageEdgeLength();
	Eigen::MatrixXd perVertexNormal();
	Eigen::MatrixXd perFaceNormal();
	Eigen::SparseMatrix<double> cotLaplace();
	Eigen::SparseMatrix<double> massMatrix(igl::MassMatrixType type = igl::MASSMATRIX_TYPE_DEFAULT);
	Eigen::SparseMatrix<double> massMatrixInv(igl::MassMatrixType type = igl::MASSMATRIX_TYPE_DEFAULT);
	PrincipalCurvature principalCurvature();
	Eigen::VectorXd gaussCurvature();
	Eigen::VectorXd meanCurvature();
	Eigen::MatrixXd baryCenter();
	Eigen::MatrixXd Grad(Eigen::VectorXd& value);
	double faceArea(int face_id); //compute area of face[face_id]
	Eigen::MatrixXd boundary(); //compute the boundary verts
	//algorithm
	void LaplaceSmooth();
	void LoopSubdiv(int number_of_subdivs = 1);
	void MarchingCubes(Eigen::VectorXd& S, Eigen::MatrixXd& GV, int nx, int ny, int nz, double isovalue);
};