#pragma once
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/property_map.h>
#include <CGAL/tags.h>
#include <CGAL/Timer.h>
#include <CGAL/Bbox_3.h>
#include <CGAL/Alpha_shape_2.h>
#include <CGAL/Alpha_shape_vertex_base_2.h>
#include <CGAL/Alpha_shape_face_base_2.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/algorithm.h>
#include <CGAL/linear_least_squares_fitting_3.h>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/random.hpp>
#include <utility>
#include "spdlog/spdlog.h"


const double PI = 3.1415926;

//typedef CGAL::Exact_predicates_exact_constructions_kernel								Kernel; //epeck
typedef CGAL::Exact_predicates_inexact_constructions_kernel								Kernel; //epick
//typedef CGAL::Simple_cartesian<double>												Kernel;  //cartesian kernel
typedef Kernel::Point_2																	Point_2;   //2D point
typedef Kernel::Segment_2																Segment_2; //2D segment
typedef Kernel::FT																		FT;
typedef Kernel::Point_3																	Point_3;   //3D point
typedef Kernel::Vector_3																Vector_3; //3D vector
typedef Kernel::Plane_3																	Plane_3; //3D plane
typedef Kernel::Iso_cuboid_3															Iso_cuboid_3; //3D cube
typedef CGAL::Bbox_3																	Bbox_3;   //3D bounding box
typedef CGAL::Point_set_3<Point_3>														Point_set_3;  //3D point set
typedef std::array<double, 3>															Color;   //rgb color
typedef Point_set_3::Property_map<Color>												Color_map;
typedef Point_set_3::Property_map<FT>													FT_map;
typedef CGAL::Surface_mesh<Point_3>														Mesh;   //mesh
typedef Mesh::Vertex_index																vertex_descriptor;
typedef Mesh::Halfedge_index															halfedge_descriptor;
typedef Mesh::Face_index																face_descriptor;
typedef CGAL::Parallel_if_available_tag													Concurrency_tag; // Concurrency
typedef boost::tuple<Point_3, int>														Point_and_int;
typedef CGAL::Search_traits_3<Kernel>													Traits_base;
typedef CGAL::Search_traits_adapter<Point_and_int,
	CGAL::Nth_of_tuple_property_map<0, Point_and_int>,
	Traits_base>																		Traits;
typedef CGAL::Orthogonal_k_neighbor_search<Traits>										K_neighbor_search;
typedef K_neighbor_search::Tree															KD_Tree;  //kd_tree
typedef CGAL::Timer																		Timer; //record time
//2d alpha shape
typedef CGAL::Alpha_shape_vertex_base_2<Kernel>											Vb;
typedef CGAL::Alpha_shape_face_base_2<Kernel>											Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb>									Tds;
typedef CGAL::Delaunay_triangulation_2<Kernel, Tds>										Triangulation_2;
typedef CGAL::Alpha_shape_2<Triangulation_2>											Alpha_shape_2;
typedef Alpha_shape_2::Alpha_shape_edges_iterator										Alpha_shape_edges_iterator;
//shape detection
typedef boost::tuple<Point_3, Vector_3, Vector_3, int>									PNNI; // points, original normal, optimized normal, points index
typedef std::vector<PNNI>																Point_vector;

//struct data
struct AlphaShapeData
{
	std::vector<Segment_2>	segments;
	double					optimal_alpha;
	AlphaShapeData(std::vector<Segment_2> m_segments, double m_optimal_alpha)
		:segments(m_segments), optimal_alpha(m_optimal_alpha)
	{}
};

struct Primitive
{
	Point_vector			points;
	Plane_3					plane;
	Vector_3				target_normal;
	Primitive(Point_vector m_points, Plane_3 m_plane)
		:points(m_points), plane(m_plane)
	{
		target_normal = Vector_3(0.0, 0.0, 0.0);
	}
};