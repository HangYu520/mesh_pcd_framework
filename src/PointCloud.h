#pragma once
#include <string>
#include <sstream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <igl/opengl/glfw/Viewer.h>
#include "Type.h"
#include "TriMesh.h"

class PointCloud
{
private:
	Point_set_3								m_pointset; //CGAL data type
	Color									m_color = Color{ 0.33, 0.66, 1.0 };  //point color (default: Blue)
	KD_Tree									kd_tree; //KDTree of point cloud
	std::vector<Primitive>					m_primitives; //detected primitives
public:
	~PointCloud() = default;
	PointCloud() = default;
	inline std::vector<std::string>			GetProperty() const { return m_pointset.properties(); }
	inline int								n_points() const { return m_pointset.size(); }
	inline Point_3							Point(int ind) const { return m_pointset.point(ind); }
	inline Vector_3							Normal(int ind) const { return m_pointset.normal(ind); }
	inline bool								Empty() const { return n_points() == 0; }
	inline void								clear() { m_pointset.clear(); kd_tree.clear(); m_primitives.clear(); }
	inline void								insert(Point_3 point) { m_pointset.insert(point); }
	void									SetColor(const Color& color);
	Eigen::MatrixXd							GetPoints() const;
	Eigen::MatrixXd							GetNormals();
	void									ReadFromFile(const char* filepath);
	void									WriteXYZ(const char* filepath);
	void									Draw(igl::opengl::glfw::Viewer& viewer) const;
	void									DrawNormal(igl::opengl::glfw::Viewer& viewer);
	void									DrawMainDir(igl::opengl::glfw::Viewer& viewer);
	void									DrawBoundary(igl::opengl::glfw::Viewer& viewer,double alpha, int sampled_points = 500);
	void									DrawAxis(igl::opengl::glfw::Viewer& viewer);
	void									DrawBoundingBox(igl::opengl::glfw::Viewer& viewer);
	void									DrawRegiongrow(igl::opengl::glfw::Viewer& viewer, double eta, int min_support);
	void									DrawRefDir(igl::opengl::glfw::Viewer& viewer);
	void									DrawOrthoDir(igl::opengl::glfw::Viewer& viewer);
	void									DrawPrallel(igl::opengl::glfw::Viewer& viewer, double tau);
	void									DrawSymm(igl::opengl::glfw::Viewer& viewer, double epsilon);
	void									DrawCoplanar(igl::opengl::glfw::Viewer& viewer, double delta);
	//algo
	Vector_3								RandomUnitNormal();
	std::vector<Color>						randomColor(int n_colors); //generate n colors randomly
	void									EstimateNormal(int n_neighbors = 20);
	Mesh									PoissonRecon();
	std::vector<Mesh>						IterPoissonRecon(double convgence_thres = 0.175);
	void									addGaussNoise(double mu = 0.0, double sigma = 1.0); //add gauss noise to points
	void									addOutliers(double ratio); //add outliers to points
	double									AverageSpacing(int n_neighbors = 10);
	void									buildKDTree();
	std::vector<int>						K_Neighbors(int query_ind, int k); //search the k nearest neighbors
	std::vector<int>						K_Neighbors(Point_3 query, int k);
	Bbox_3									BoundingBox(); //return bounding box
	void									Project_along_x(); //x-components of points are set to 0
	void									Project_along_y(); //y-components of points are set to 0
	void									Project_along_z(); //z-components of points are set to 0
	void									scale(double ratio); //scale the point cloud
	void									removeDuplicatePoints();//remove duplicate points in the point cloud
	AlphaShapeData							alpha_shape_2d(double alpha = 3); //return 2d alpha shape segments
	std::vector<Segment_2>					connect_segments(std::vector<Segment_2>& segments);
	std::vector<Point_2>					downsample(std::vector<Segment_2>& connect_segments, int sampled_points = 500);
	std::vector<Primitive>					improved_region_grow(double eta, int min_support); //eta : angle (deg) threshold between original normals
	std::vector<Vector_3>					reference_direction();
	std::vector<Vector_3>					orthogonal_direction(const std::vector<Vector_3>& ref_dirs);
	void									set_parallel(std::vector<Primitive>& primitives, std::vector<Vector_3>& ref_dirs, std::vector<Vector_3>& ortho_dirs, double tau); //tau: angle (deg) threshold between face normals
	void									set_symmetry(std::vector<Primitive>& primitives, std::vector<Vector_3>& ortho_dirs, double epsilon); //epsilon : threshold to detect symmetry
	void									set_coplanar(std::vector<Primitive>& primitives, double delta); // delta : distance threshold to detect coplanar
	Primitive								merge_primives(const std::vector<Primitive>& primitives); //return the merged primitive
private:
	Eigen::MatrixXd							GetColors() const;
};