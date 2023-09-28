#include "PointCloud.h"
#include <CGAL/IO/read_points.h>
#include <CGAL/IO/write_xyz_points.h>
#include <CGAL/jet_estimate_normals.h>
#include <CGAL/pca_estimate_normals.h>
#include <CGAL/mst_orient_normals.h>
#include <CGAL/poisson_surface_reconstruction.h>
#include <CGAL/compute_average_spacing.h>
#include <CGAL/point_generators_3.h>
#include <CGAL/bounding_box.h>

//internal functions used for algo
static double GaussFunc(double x, double sig)
{
    //exp{-(x/sig)^2}
    return exp(-x * x / (sig * sig));
}

static double DegToRad(double degree)
{
    return degree * PI / 180;
}

static double RadToDeg(double radian)
{
    return radian * 180 / PI;
}

static double angle(const Vector_3& v1, const Vector_3& v2, bool nodirection)
{
    //if nodirection : angle range from [0,90], else [0,180]
    double dot_product = v1 * v2;
    double v1_length = std::sqrt(v1.squared_length());
    double v2_length = std::sqrt(v2.squared_length());

    double cos_angle = dot_product / (v1_length * v2_length);
    double angle = std::acos(cos_angle);

    if (nodirection && angle > PI / 2)
        angle = PI - angle;

    return angle;
}

static bool exsist(Vector_3& v, std::vector<Vector_3>& arr)
{
    for (int i = 0; i < arr.size(); i++)
    {
        if (arr[i] == v)
            return true;
    }

    return false;
}

static double dot_vector(const Vector_3& v1, const Vector_3& v2)
{
    return v1.x() * v2.x() + v1.y() * v2.y() + v1.z() * v2.z();
}

std::array<Eigen::Matrix3d, 6> getRotationMat(double alpha, double beta, double zeta)
{
    //return the three rotation matrix and their derivative (R1, R2, R3, dR1, dR2, dR3)
    //alpha beta zeta : rad

    Eigen::Matrix3d R1, R2, R3, dR1, dR2, dR3;

    R1 << 1, 0, 0,
        0, std::cos(alpha), std::sin(alpha),
        0, -std::sin(alpha), std::cos(alpha);

    R2 << std::cos(beta), 0, -std::sin(beta),
        0, 1, 0,
        std::sin(beta), 0, std::cos(beta);

    R3 << std::cos(zeta), std::sin(zeta), 0,
        -std::sin(zeta), std::cos(zeta), 0,
        0, 0, 1;

    dR1 << 0, 0, 0,
        0, -std::sin(alpha), std::cos(alpha),
        0, -std::cos(alpha), -std::sin(alpha);

    dR2 << -std::sin(beta), 0, -std::cos(beta),
        0, 0, 0,
        std::cos(beta), 0, -std::sin(beta);

    dR3 << -std::sin(zeta), std::cos(zeta), 0,
        -std::cos(zeta), -std::sin(zeta), 0,
        0, 0, 0;

    std::array<Eigen::Matrix3d, 6> result = { R1, R2, R3, dR1, dR2, dR3 };
    return result;
}

template <class OutputIterator>
void alpha_edges(const Alpha_shape_2& A, OutputIterator out)
{
    Alpha_shape_edges_iterator it = A.alpha_shape_edges_begin(),
        end = A.alpha_shape_edges_end();
    for (; it != end; ++it)
        *out++ = A.segment(*it);
}

void savePointCloudToXYZ(const Eigen::MatrixXd& P, const Eigen::MatrixXd& C, const std::string& filename)
{
    assert(P.rows() == C.rows() && P.cols() == 3 && C.cols() == 3);
    std::ofstream ofs(filename, std::ios_base::out);
    if (!ofs.is_open()) {
        spdlog::error("Failed to open file {} for writing", filename);
        return;
    }
    for (int i = 0; i < P.rows(); ++i) {
        ofs << std::setprecision(12) << P(i, 0) << " " << P(i, 1) << " " << P(i, 2) << " " <<
            C(i, 0) * 255 << " " << C(i, 1) * 255 << " " << C(i, 2) * 255 << std::endl;
    }
    ofs.close();
    spdlog::info("points are saved to {}", filename);
}

std::vector<Segment_2> deleterepeatpoints(std::vector<Segment_2>& connect_segments)
{
    std::vector<Segment_2> unique_segments;
    std::unordered_set<Point_2> visted_points;
    for (int i = 0; i < connect_segments.size(); i++)
    {
        if (!visted_points.count(connect_segments[i].source()))
        {
            unique_segments.push_back(connect_segments[i]);
            visted_points.insert(connect_segments[i].source());
        }
    }
    return unique_segments;
}

std::vector<Segment_2> deleteshortSegments(std::vector<Segment_2>& connect_segments)
{
    //delete segments which are too short
    double eps = 1e-6;
    std::vector<Segment_2> deleted_segments;
    for (int i = 0; i < connect_segments.size(); i++)
    {
        if (connect_segments[i].squared_length() > eps)
            deleted_segments.push_back(connect_segments[i]);
    }
    return deleted_segments;
}

std::vector<Point_2> PointCloud::downsample(std::vector<Segment_2>& connect_segments, int sampled_points)
{
    std::vector<Point_2> downsample_points;
 
    //length-based downsample
    std::vector<double> segments_length;
    double sum_length = 0;
    for (int i = 0; i < connect_segments.size(); i++)
    {
        sum_length += CGAL::sqrt(connect_segments[i].squared_length());
        segments_length.push_back(CGAL::sqrt(connect_segments[i].squared_length()));
    }
    std::vector<double> points_attrib;
    points_attrib.push_back(0);
    for (int i = 1; i < segments_length.size(); i++)
    {
        points_attrib.push_back(segments_length[i - 1] + points_attrib[i - 1]);
    }
    double per_length = sum_length / double(sampled_points);
    double iter = 0;
    int id = 0;
    int trigger = 0; //breakpoint
    while (iter < sum_length)
    {
        bool found = false;
        while (!found && id < points_attrib.size() - 1)
        {
            if (points_attrib[id] > iter)
                id--;
            if (points_attrib[id] <= iter && points_attrib[id + 1] > iter)
            {
                found = true;
                trigger++;
            }
            id++;
        }
        if (found)
        {
            double t1 = iter - points_attrib[id - 1];
            double t2 = points_attrib[id] - iter;
            double t = t2 / (t1 + t2);
            Point_2 point(t * connect_segments[id - 1].source().x()
                + (1 - t) * connect_segments[id - 1].target().x(),
                t * connect_segments[id - 1].source().y()
                + (1 - t) * connect_segments[id - 1].target().y());
            downsample_points.push_back(point);
        }
        iter += per_length;
    }

    return downsample_points;
}

void PointCloud::SetColor(const Color& color)
{
    m_color = color;
}

void PointCloud::ReadFromFile(const char* filepath)
{
	m_pointset.add_normal_map();
    if (!CGAL::IO::read_points(filepath, m_pointset.index_back_inserter(),
        CGAL::parameters::point_map(m_pointset.point_push_map())
        .normal_map(m_pointset.normal_push_map())))
    {
        spdlog::error("Can't read input file {}", filepath);
    }
}

void PointCloud::WriteXYZ(const char* filepath)
{
    if (CGAL::IO::write_XYZ(filepath, m_pointset))
        spdlog::info("save point cloud to {} successfully!", filepath);
    else
        spdlog::error("failed to save file {}", filepath);
}

Eigen::MatrixXd PointCloud::GetColors() const
{
    int n = n_points();
    Eigen::MatrixXd C(n, 3);
    for (int i = 0; i < n; i++)
    {
        C(i, 0) = m_color.at(0);
        C(i, 1) = m_color.at(1);
        C(i, 2) = m_color.at(2);
    }
    return C;
}

Eigen::MatrixXd PointCloud::GetPoints() const
{
    int n = n_points();
    Eigen::MatrixXd P(n, 3);
    int i = 0;
    for (auto iter = m_pointset.begin(); iter != m_pointset.end(); iter++)
    {
        Point_3 point = m_pointset.point(*iter);
        P(i, 0) = point.x();
        P(i, 1) = point.y();
        P(i, 2) = point.z();
        i++;
    }
    return P;
}

Eigen::MatrixXd PointCloud::GetNormals()
{
    int n = n_points();
    Eigen::MatrixXd N(n, 3);
    int i = 0;
    for (auto iter = m_pointset.begin(); iter != m_pointset.end(); iter++)
    {
        Vector_3 normal = m_pointset.normal(*iter);
        N(i, 0) = normal.x();
        N(i, 1) = normal.y();
        N(i, 2) = normal.z();
        i++;
    }
    return N;
}

void PointCloud::Draw(igl::opengl::glfw::Viewer& viewer) const
{
    if (Empty())
        return;
    viewer.data().clear_points();
    Eigen::MatrixXd P = GetPoints();
    Eigen::MatrixXd C = GetColors();
    viewer.data().set_points(P, C);
    viewer.data().point_size = 4.0;
    viewer.core().align_camera_center(P);
}

void PointCloud::DrawNormal(igl::opengl::glfw::Viewer& viewer)
{
    if (m_pointset.normals().empty())
        EstimateNormal(30);
    double ratio = AverageSpacing();
    Eigen::MatrixXd P = GetPoints();
    Eigen::MatrixXd N = GetNormals();
    SetColor(Color{ 1,0,0 });
    Eigen::MatrixXd C = GetColors();
    SetColor(Color{ 0,0,1 });
    Eigen::MatrixXd E = P + ratio * N;
    viewer.data().add_edges(P, E, C);
}

void PointCloud::DrawMainDir(igl::opengl::glfw::Viewer& viewer)
{
    //bounding box
    Bbox_3 bbx = BoundingBox();
    Eigen::MatrixXd C(3, 3), E(3, 3), CL(3, 3);
    C(0, 0) = (bbx.xmin() + bbx.xmax()) / 2;
    C(0, 1) = (bbx.ymin() + bbx.ymax()) / 2;
    C(0, 2) = (bbx.zmin() + bbx.zmax()) / 2;
    std::cout << "center: \n" << C.row(0) << std::endl;
    C.row(1) = C.row(0);
    C.row(2) = C.row(0);
    CL << 1, 0, 0,
        1, 0, 0,
        1, 0, 0;
    double length = std::max(std::max(bbx.xmax() - bbx.xmin(),
        bbx.ymax() - bbx.ymin()),
        bbx.zmax() - bbx.zmin()) / 2;
    //need input main directions
    spdlog::info("Please input the first direction: ");
    std::cin >> E(0, 0) >> E(0, 1) >> E(0, 2);
    spdlog::info("Please input the second direction: ");
    std::cin >> E(1, 0) >> E(1, 1) >> E(1, 2);
    spdlog::info("Please input the third direction: ");
    std::cin >> E(2, 0) >> E(2, 1) >> E(2, 2);
    E = C + length * E;
    int width = 1;
    for (int i = 0; i < width; i++)
        viewer.data().add_edges(C, E, CL);
}

void PointCloud::DrawBoundary(igl::opengl::glfw::Viewer& viewer, double alpha, int sampled_points)
{
    //projection first
    viewer.data().clear_points();
    AlphaShapeData asd = alpha_shape_2d(alpha);
    spdlog::info("recompute with alpha = {}", asd.optimal_alpha * 10);
    AlphaShapeData asd2 = alpha_shape_2d(asd.optimal_alpha * 10);
    std::vector<Segment_2> segments = asd2.segments;
    std::vector<Segment_2> c_segments = connect_segments(segments);
    //print info
    spdlog::info("num of boundary points: {}", c_segments.size());
    //delete short segments
    //std::vector<Segment_2> nsc_segments = deleteshortSegments(c_segments); // no short connect segments
    //std::cout << "num of deleted boundary points: " << nsc_segments.size() << std::endl;
    //downsample segments
    std::vector<Point_2> d_points = downsample(c_segments, sampled_points);
    //delete repeat source points in segments
    //std::vector<Segment_2> u_segments = deleterepeatpoints(c_segments);
    //print info
    spdlog::info("num of sampled boundary points: {}", d_points.size());
    m_pointset.clear();
    //store in Eigen matrix
    Eigen::MatrixXd I_P(c_segments.size(), 3), I_C(c_segments.size(), 3); //initial boundary points
    for (int i = 0; i < c_segments.size(); i++)
    {
        I_P.row(i) << c_segments[i].source().x(), c_segments[i].source().y(), 0;
        I_C.row(i) << 1, 0, 0;
    }
    Eigen::MatrixXd S_P(d_points.size(), 3), S_C(d_points.size(), 3); //sampled boundary points
    for (int i = 0; i < d_points.size(); i++)
    {
        S_P.row(i) << d_points[i].x(), d_points[i].y(), 0;
        S_C.row(i) << 1, 0, 0;
        m_pointset.insert(Point_3(d_points[i].x(), d_points[i].y(), 0));
    }
    savePointCloudToXYZ(I_P, I_C, "res/pcd/car/boundary_points.xyz"); //save initial boundary points
    savePointCloudToXYZ(S_P, S_C, "res/pcd/car/boundary.xyz"); //save sampled boundary points
    //viewer.data().set_points(S_P, S_C);
}

void PointCloud::DrawAxis(igl::opengl::glfw::Viewer& viewer)
{
    Bbox_3 bbx = BoundingBox();
    Eigen::MatrixXd P1(3, 3), P2(3, 3), C(3, 3);
    Point_3 center((bbx.xmin() + bbx.xmax()) / 2,
        (bbx.ymin() + bbx.ymax()) / 2,
        (bbx.zmin() + bbx.zmax()) / 2);
    double x_length = bbx.xmax() - bbx.xmin();
    double y_length = bbx.ymax() - bbx.ymin();
    double z_length = bbx.zmax() - bbx.zmin();
    P1.row(0) << center.x(), center.y(), center.z();
    P1.row(1) = P1.row(0);
    P1.row(2) = P1.row(0);
    P2 << x_length, 0, 0,
        0, y_length, 0,
        0, 0, z_length;
    P2 = P2 * 0.6 + P1;
    C << 1, 0, 0,
        0, 1, 0,
        0, 0, 1;
    viewer.data().add_edges(P1, P2, C);
    viewer.data().add_label(P2.row(0), "X");
    viewer.data().add_label(P2.row(1), "Y");
    viewer.data().add_label(P2.row(2), "Z");
}

void PointCloud::DrawBoundingBox(igl::opengl::glfw::Viewer& viewer)
{
    Bbox_3 bbx = BoundingBox();
    Eigen::MatrixXd C(1, 3);
    C << 0.5, 0, 0.5;
    std::vector<Eigen::MatrixXd> corners;
    double x[2] = { bbx.xmin(),bbx.xmax() };
    double y[2] = { bbx.ymin(),bbx.ymax() };
    double z[2] = { bbx.zmin(),bbx.zmax() };
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
            {
                Eigen::MatrixXd point(1, 3);
                point << x[i], y[j], z[k];
                corners.push_back(point);
            }
    viewer.data().add_edges(corners[0], corners[4], C);
    viewer.data().add_edges(corners[1], corners[5], C);
    viewer.data().add_edges(corners[3], corners[7], C);
    viewer.data().add_edges(corners[2], corners[6], C);
    viewer.data().add_edges(corners[0], corners[2], C);
    viewer.data().add_edges(corners[1], corners[3], C);
    viewer.data().add_edges(corners[0], corners[4], C);
    viewer.data().add_edges(corners[4], corners[6], C);
    viewer.data().add_edges(corners[5], corners[7], C);
    viewer.data().add_edges(corners[0], corners[1], C);
    viewer.data().add_edges(corners[2], corners[3], C);
    viewer.data().add_edges(corners[4], corners[5], C);
    viewer.data().add_edges(corners[6], corners[7], C);

    std::stringstream labelx, labely, labelz;
    labelx << bbx.xmax() - bbx.xmin();
    viewer.data().add_label(((corners[0] + corners[4]) / 2).row(0), labelx.str());
    labely << bbx.ymax() - bbx.ymin();
    viewer.data().add_label(((corners[4] + corners[6]) / 2).row(0), labely.str());
    labelz << bbx.zmax() - bbx.zmin();
    viewer.data().add_label(((corners[4] + corners[5]) / 2).row(0), labelz.str());
}

void PointCloud::DrawRegiongrow(igl::opengl::glfw::Viewer& viewer, double eta, int min_support)
{
    std::vector<Primitive> primitives = improved_region_grow(eta, min_support);
    std::vector<Color> colors = randomColor(primitives.size());
    if (Empty())
        return;
    viewer.data().clear_points();
    m_pointset.clear();
    m_pointset.add_normal_map();
    std::vector<Color> point_colors;
    //asign color
    for (int i = 0; i < primitives.size(); i++)
    {
        Primitive primitive = primitives[i];
        for (int j = 0; j < primitive.points.size(); j++)
        {
            Point_3 p = primitive.points[j].get<0>();
            Vector_3 v = primitive.points[j].get<2>();
            //Vector_3 v = Vector_3(primitive.plane.a(), primitive.plane.b(), primitive.plane.c());
            m_pointset.insert(p, v);
            point_colors.push_back(colors[i]);
        }
    }
    Eigen::MatrixXd P = GetPoints();
    Eigen::MatrixXd C = GetColors();
    for (int i = 0; i < point_colors.size(); i++)
        C.row(i) << point_colors[i][0], point_colors[i][1], point_colors[i][2];
    viewer.data().set_points(P, C);
    viewer.data().point_size = 4.0;
    viewer.core().align_camera_center(P);
}

void PointCloud::DrawRefDir(igl::opengl::glfw::Viewer& viewer)
{
    std::vector<Vector_3> ref_dirs = reference_direction();
    Bbox_3 bbx = BoundingBox();
    Eigen::MatrixXd P1(3, 3), P2(3, 3), C(3, 3);
    Point_3 center((bbx.xmin() + bbx.xmax()) / 2,
        (bbx.ymin() + bbx.ymax()) / 2,
        (bbx.zmin() + bbx.zmax()) / 2);
    double x_length = bbx.xmax() - bbx.xmin();
    double y_length = bbx.ymax() - bbx.ymin();
    double z_length = bbx.zmax() - bbx.zmin();
    double L = 0.5 * sqrt(x_length * x_length + y_length * y_length + z_length * z_length);
    P1.row(0) << center.x(), center.y(), center.z();
    P1.row(1) = P1.row(0);
    P1.row(2) = P1.row(0);
    Point_3 p1 = center + L * ref_dirs[0];
    P2.row(0) << p1.x(), p1.y(), p1.z();
    Point_3 p2 = center + L * ref_dirs[1];
    P2.row(1) << p2.x(), p2.y(), p2.z();
    Point_3 p3 = center + L * ref_dirs[2];
    P2.row(2) << p3.x(), p3.y(), p3.z();
    C << 1, 0, 0,
        0, 1, 0,
        0, 0, 1;
    viewer.data().add_edges(P1, P2, C);
    viewer.data().add_label(P2.row(0), "r1");
    viewer.data().add_label(P2.row(1), "r2");
    viewer.data().add_label(P2.row(2), "r3");
}

void PointCloud::DrawOrthoDir(igl::opengl::glfw::Viewer& viewer)
{
    std::vector<Vector_3> ref_dirs = reference_direction();
    std::vector<Vector_3> ortho_dirs = orthogonal_direction(ref_dirs);
    Bbox_3 bbx = BoundingBox();
    Eigen::MatrixXd P1(3, 3), P2(3, 3), C(3, 3);
    Point_3 center((bbx.xmin() + bbx.xmax()) / 2,
        (bbx.ymin() + bbx.ymax()) / 2,
        (bbx.zmin() + bbx.zmax()) / 2);
    double x_length = bbx.xmax() - bbx.xmin();
    double y_length = bbx.ymax() - bbx.ymin();
    double z_length = bbx.zmax() - bbx.zmin();
    double L = 0.5 * sqrt(x_length * x_length + y_length * y_length + z_length * z_length);
    P1.row(0) << center.x(), center.y(), center.z();
    P1.row(1) = P1.row(0);
    P1.row(2) = P1.row(0);
    Point_3 p1 = center + L * ortho_dirs[0];
    P2.row(0) << p1.x(), p1.y(), p1.z();
    Point_3 p2 = center + L * ortho_dirs[1];
    P2.row(1) << p2.x(), p2.y(), p2.z();
    Point_3 p3 = center + L * ortho_dirs[2];
    P2.row(2) << p3.x(), p3.y(), p3.z();
    C << 1, 0, 0,
        0, 1, 0,
        0, 0, 1;
    viewer.data().add_edges(P1, P2, C);
    viewer.data().add_label(P2.row(0), "r1*");
    viewer.data().add_label(P2.row(1), "r2*");
    viewer.data().add_label(P2.row(2), "r3*");
}

void PointCloud::DrawPrallel(igl::opengl::glfw::Viewer& viewer, double tau)
{
    //required : already running DrawRegiongrow
    std::vector<Vector_3> ref_dirs = reference_direction();
    std::vector<Vector_3> ortho_dirs = orthogonal_direction(ref_dirs);
    set_parallel(m_primitives, ref_dirs, ortho_dirs, tau);
    if (Empty())
        return;
    viewer.data().clear_points();
    m_pointset.clear();
    m_pointset.add_normal_map();
    std::vector<Color> point_colors;
    //asign color
    for (int i = 0; i < m_primitives.size(); i++)
    {
        Primitive primitive = m_primitives[i];
        for (int j = 0; j < primitive.points.size(); j++)
        {
            Point_3 p = primitive.points[j].get<0>();
            Vector_3 v = primitive.points[j].get<2>();
            //Vector_3 v = Vector_3(primitive.plane.a(), primitive.plane.b(), primitive.plane.c());
            m_pointset.insert(p, v);
            if (primitive.target_normal == ortho_dirs[0])
                point_colors.push_back(Color{ 1.0, 0.0, 0.0 });
            else if (primitive.target_normal == ortho_dirs[1])
                point_colors.push_back(Color{ 0.0, 1.0, 0.0 });
            else if (primitive.target_normal == ortho_dirs[2])
                point_colors.push_back(Color{ 0.0, 0.0, 1.0 });
            else if (primitive.target_normal == Vector_3(0.0, 0.0, 0.0))
                point_colors.push_back(Color{ 0.33, 0.66, 1.0 });
        }
    }
    Eigen::MatrixXd P = GetPoints();
    Eigen::MatrixXd C = GetColors();
    for (int i = 0; i < point_colors.size(); i++)
        C.row(i) << point_colors[i][0], point_colors[i][1], point_colors[i][2];
    viewer.data().set_points(P, C);
    viewer.data().point_size = 4.0;
    viewer.core().align_camera_center(P);
}

void PointCloud::DrawSymm(igl::opengl::glfw::Viewer& viewer, double epsilon)
{
    //required : already running DrawRegiongrow and setparallel
    std::vector<Vector_3> ref_dirs = reference_direction();
    std::vector<Vector_3> ortho_dirs = orthogonal_direction(ref_dirs);
    set_symmetry(m_primitives, ortho_dirs, epsilon);
    Bbox_3 bbx = BoundingBox();
    Point_3 center((bbx.xmin() + bbx.xmax()) / 2,
        (bbx.ymin() + bbx.ymax()) / 2,
        (bbx.zmin() + bbx.zmax()) / 2);
    double x_length = bbx.xmax() - bbx.xmin();
    double y_length = bbx.ymax() - bbx.ymin();
    double z_length = bbx.zmax() - bbx.zmin();
    double L = 0.1 * sqrt(x_length * x_length + y_length * y_length + z_length * z_length);
    std::vector<Point_3> starts, ends;
    std::vector<Vector_3> normals;
    for (Primitive primitive : m_primitives)
    {
        if (primitive.target_normal != Vector_3(0.0, 0.0,0.0) && !exsist(primitive.target_normal, ortho_dirs))
        {
            starts.push_back(primitive.points[0].get<0>());
            Vector_3 n = primitive.target_normal;
            if (angle(primitive.points[0].get<1>(), primitive.target_normal, false) > PI / 2)
                n = -1 * n;
            ends.push_back(primitive.points[0].get<0>() + L * n);
            normals.push_back(primitive.target_normal);
        }
    }
    Eigen::MatrixXd P1(starts.size(), 3), P2(ends.size(), 3), C(starts.size(), 3);
    for (int i = 0; i < starts.size(); i++)
    {
        P1.row(i) << starts[i][0], starts[i][1], starts[i][2];
        P2.row(i) << ends[i][0], ends[i][1], ends[i][2];
        C.row(i) << 0, 0, 0;
        std::stringstream ss;
        ss << "(" << normals[i][0] << "," << normals[i][1] << "," << normals[i][2] << ")";
        viewer.data().add_label(P2.row(i), ss.str());
    }
    viewer.data().add_edges(P1, P2, C);
}

void PointCloud::DrawCoplanar(igl::opengl::glfw::Viewer& viewer, double delta)
{
    //required : already running DrawRegiongrow
    set_coplanar(m_primitives, delta);
    std::vector<Color> colors = randomColor(m_primitives.size());
    if (Empty())
        return;
    viewer.data().clear_points();
    m_pointset.clear();
    m_pointset.add_normal_map();
    std::vector<Color> point_colors;
    //asign color
    for (int i = 0; i < m_primitives.size(); i++)
    {
        Primitive primitive = m_primitives[i];
        for (int j = 0; j < primitive.points.size(); j++)
        {
            Point_3 p = primitive.points[j].get<0>();
            Vector_3 v = primitive.points[j].get<2>();
            //Vector_3 v = Vector_3(primitive.plane.a(), primitive.plane.b(), primitive.plane.c());
            m_pointset.insert(p, v);
            point_colors.push_back(colors[i]);
        }
    }
    Eigen::MatrixXd P = GetPoints();
    Eigen::MatrixXd C = GetColors();
    for (int i = 0; i < point_colors.size(); i++)
        C.row(i) << point_colors[i][0], point_colors[i][1], point_colors[i][2];
    viewer.data().set_points(P, C);
    viewer.data().point_size = 4.0;
    viewer.core().align_camera_center(P);
}

void PointCloud::DrawOptPrim(igl::opengl::glfw::Viewer& viewer)
{
    primitive_optimize(m_primitives, std::array<double, 3>{0.0, 0.0, 0.0});
    ProjectPrim(m_primitives);
    if (Empty())
        return;
    viewer.data().clear_points();
    m_pointset.clear();
    m_pointset.add_normal_map();
    std::vector<Color> point_colors;
    //asign color
    for (int i = 0; i < m_primitives.size(); i++)
    {
        Primitive primitive = m_primitives[i];
        for (int j = 0; j < primitive.points.size(); j++)
        {
            Point_3 p = primitive.points[j].get<0>();
            Vector_3 v = primitive.points[j].get<2>();
            //Vector_3 v = Vector_3(primitive.plane.a(), primitive.plane.b(), primitive.plane.c());
            m_pointset.insert(p, v);
            point_colors.push_back(Color{ 0.33, 0.66, 1.0 });
        }
    }
    Eigen::MatrixXd P = GetPoints();
    Eigen::MatrixXd C = GetColors();
    for (int i = 0; i < point_colors.size(); i++)
        C.row(i) << point_colors[i][0], point_colors[i][1], point_colors[i][2];
    viewer.data().set_points(P, C);
    viewer.data().point_size = 4.0;
    viewer.core().align_camera_center(P);
}

Vector_3 PointCloud::RandomUnitNormal()
{
    //return a unit normal randomly
    Point_3 p;
    CGAL::Random_points_on_sphere_3<Point_3> g(1);
    p = *g;
    return Vector_3(p.x(), p.y(), p.z());
}

std::vector<Color> PointCloud::randomColor(int n_colors)
{
    std::vector<Color> colors;
    boost::mt19937 zgy;  //Uniform pseudorandom number generator
    zgy.seed(static_cast<unsigned int>(time(0)));	//random seed
    boost::uniform_real<> ur(0, 1);
    for (int i = 0; i < n_colors; i++)
    {
        Color color = Color{ ur(zgy), ur(zgy), ur(zgy) };
        colors.push_back(color);
    }
    return colors;
}

void PointCloud::EstimateNormal(int n_neighbors)
{
    Timer timer;
    timer.start();
    spdlog::info("Start Estimate Normals, n_neighbors: {}", n_neighbors);
    //CGAL::jet_estimate_normals<Concurrency_tag>
    //    (m_pointset, n_neighbors);
    CGAL::pca_estimate_normals<Concurrency_tag>
        (m_pointset, n_neighbors);
      // Orientation of normals, returns iterator to first unoriented point
    typename Point_set_3::iterator unoriented_points_begin =
        CGAL::mst_orient_normals(m_pointset, n_neighbors);
    m_pointset.remove(unoriented_points_begin, m_pointset.end());
    spdlog::info("Done. Time: {} s", timer.time());
}

double PointCloud::AverageSpacing(int n_neighbors)
{
    FT average_spacing = CGAL::compute_average_spacing<Concurrency_tag>(
        m_pointset, n_neighbors);
    return average_spacing;
}

Mesh PointCloud::PoissonRecon()
{
    Timer timer;
    timer.start();
    spdlog::info("Start Poisson Reconstruction.");
    Mesh output_mesh;
    if (!m_pointset.has_normal_map())
    {
        spdlog::warn("Please Estimate Normal First.");
        return output_mesh;
    }
    double spacing = AverageSpacing();
    CGAL::poisson_surface_reconstruction_delaunay
    (m_pointset.begin(), m_pointset.end(),
        m_pointset.point_map(), m_pointset.normal_map(),
        output_mesh, spacing);
    spdlog::info("Done. Time: {} s", timer.time());
    return output_mesh;
}

std::vector<Mesh> PointCloud::IterPoissonRecon(double convgence_thres)
{
    Timer timer;
    timer.start();
    //implementaion of iPSR, TOG, 2022
    //return a mesh vector of each iter steps
    spdlog::info("Start iterative Poisson Reconstruction. convergence_thres: {}", convgence_thres);
    std::vector<Mesh> mesh_arr;
    //build kdtree for seaching neighbors
    buildKDTree();
    //initialize a normal for each point randomly
    for (int i = 0; i < n_points(); i++)
        m_pointset.normal(i) = RandomUnitNormal();
    //iteration
    int MaxIter = 30;
    for (int i = 0; i < MaxIter; i++)
    {
        timer.reset();
        spdlog::info("iter: {}", i + 1);
        std::priority_queue<double> normal_differ; //normal difference with last iteration
        std::unordered_map<int, std::vector<int>> face_list;
        std::unordered_map<int, double> facearea_list;
        Mesh m = PoissonRecon(); //poisson reconstruction
        mesh_arr.push_back(m);
        TriMesh mesh;
        mesh.SetMesh(m);
        Eigen::MatrixXd face_normal = mesh.perFaceNormal();
        Eigen::MatrixXd bary_center = mesh.baryCenter();
        spdlog::info("construct face list,faces: {}", mesh.n_faces());
        for (int j = 0; j < mesh.n_faces(); j++)
        {
            facearea_list[j] = mesh.faceArea(j);
            Point_3 p(bary_center(j, 0), bary_center(j, 1), bary_center(j, 2));
            std::vector<int> neighbors = K_Neighbors(p, 15);
            for (int k : neighbors)
                face_list[k].push_back(j);
        }
        spdlog::info("update normal");
        //update the normal of each point
        for (int j = 0; j < n_points(); j++)
        {
            if (face_list.find(j) == face_list.end())
                continue;
            Vector_3 v_new(0, 0, 0);
            for (int k = 0; k < face_list[j].size(); k++)
            {
                int fid = face_list[j][k];
                Vector_3 vn(face_normal(fid, 0), face_normal(fid, 1), face_normal(fid, 2));
                v_new += facearea_list[fid] * vn;
            }
            //normalize
            double len = sqrt(v_new.squared_length());
            v_new /= len;
            //compute error
            double err = (1 - CGAL::scalar_product(m_pointset.normal(j), v_new)) / 2;
            normal_differ.push(err);
            //update
            m_pointset.normal(j) = v_new;
        }
        //if convergence, break
        int size = n_points() > 1000 ? n_points() / 1000 : 10;
        double averr = 0;
        for (int j = 0; j < size; j++)
        {
            averr += normal_differ.top();
            normal_differ.pop();
        }
        averr /= size;
        spdlog::info("average error: {} time: {} s", averr, timer.time());
        if (averr < convgence_thres)
        {
            spdlog::info("Convergence");
            break;
        }
    }
    spdlog::info("Done");

    return mesh_arr;
}

void PointCloud::addGaussNoise(double mu, double sigma)
{
    boost::mt19937 zgy;  //Uniform pseudorandom number generator
    zgy.seed(static_cast<unsigned int>(time(0)));	//random seed
    boost::normal_distribution<> nd(mu, sigma);    //normal(Gauss) distribution
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<>> gauss_noise(zgy, nd);
    
    // add noise
    for (int i = 0; i < n_points(); i++)
    {
        Vector_3 noises(static_cast<double> (gauss_noise()), 
                       static_cast<double> (gauss_noise()), 
                       static_cast<double> (gauss_noise()));
        m_pointset.point(i) += noises;
    }
}

void PointCloud::addOutliers(double ratio)
{
    //ratio: percentage of outliers, final number of points is (1+ratio)*nums
    //outliers array
    std::vector<Point_3> outliers;
    int n_outliers = std::floor(n_points() * ratio);
    //get bounding box of the point cloud
    Bbox_3 bbx = BoundingBox();
    //uniform int distribution sampling
    boost::mt19937 zgy;  //Uniform pseudorandom number generator
    zgy.seed(static_cast<unsigned int>(time(0)));	//random seed
    boost::uniform_real<> urx(bbx.xmin(), bbx.xmax()),
        ury(bbx.ymin(), bbx.ymax()),
        urz(bbx.zmin(), bbx.zmax());
    buildKDTree(); //build kdtree for neighbor search
    //randomly sample outliers in bounding box
    spdlog::info("generating outliers");
    while (n_outliers > 0)
    {
        Point_3 outlier(urx(zgy), ury(zgy), urz(zgy));
        //outlier must outside the point cloud
        //find the nearst point in the point cloud
        std::vector<int> nearstpoint = K_Neighbors(outlier, 1);
        Vector_3 dir = Point(nearstpoint[0]) - outlier;
        if (CGAL::scalar_product(Normal(nearstpoint[0]), dir) > 0)
        {
            outliers.push_back(outlier);
            n_outliers--;
        }
    }
    //add outliers to point set
    spdlog::info("insert outliers");
    for (Point_3 outlier : outliers)
    {
        m_pointset.insert(outlier);
    }
}

void PointCloud::buildKDTree()
{
    //build kdTree for point cloud
    std::vector<Point_3> points;
    std::vector<int> indices;
    for (int i = 0; i < n_points(); i++)
    {
        points.push_back(m_pointset.point(i));
        indices.push_back(i);
    }
    kd_tree.insert(boost::make_zip_iterator(boost::make_tuple(points.begin(), indices.begin())),
        boost::make_zip_iterator(boost::make_tuple(points.end(), indices.end())));
}

std::vector<int> PointCloud::K_Neighbors(int query_ind, int k)
{
    //find the k-nearest points of point[query_ind]
    std::vector<int> results;
    Point_3 query = m_pointset.point(query_ind);
    K_neighbor_search search(kd_tree, query, k);
    for (auto iter = search.begin(); iter != search.end(); iter++)
        results.push_back(boost::get<1>(iter->first));
    return results;
}

std::vector<int> PointCloud::K_Neighbors(Point_3 query, int k)
{
    //find the k-nearest points(in pointcloud) of a specified point
    std::vector<int> results;
    K_neighbor_search search(kd_tree, query, k);
    for (auto iter = search.begin(); iter != search.end(); iter++)
        results.push_back(boost::get<1>(iter->first));
    return results;
}

Bbox_3 PointCloud::BoundingBox()
{
    std::vector<Point_3> points;
    for (int i = 0; i < n_points(); i++)
        points.push_back(m_pointset.point(i));
    Iso_cuboid_3 cuboid = CGAL::bounding_box(points.begin(), points.end());
    return cuboid.bbox();
}

void PointCloud::Project_along_x()
{
    for (int i = 0; i < n_points(); i++)
    {
        Point_3 point(m_pointset.point(i).y(), m_pointset.point(i).z(), 0);
        m_pointset.point(i) = point;
    }
}

void PointCloud::Project_along_y()
{
    for (int i = 0; i < n_points(); i++)
    {
        Point_3 point(m_pointset.point(i).x(), m_pointset.point(i).z(), 0);
        m_pointset.point(i) = point;
    }
}

void PointCloud::Project_along_z()
{
    for (int i = 0; i < n_points(); i++)
    {
        Point_3 point(m_pointset.point(i).x(), m_pointset.point(i).y(), 0);
        m_pointset.point(i) = point;
    }
}

void PointCloud::scale(double ratio)
{
    Bbox_3 bbx = BoundingBox();
    Point_3 center((bbx.xmin() + bbx.xmax()) / 2,
        (bbx.ymin() + bbx.ymax()) / 2,
        (bbx.zmin() + bbx.zmax()) / 2);
    for (int i = 0; i < n_points(); i++)
    {
        m_pointset.point(i) = center + ratio * (m_pointset.point(i) - center);
    }
}

void PointCloud::removeDuplicatePoints()
{
    std::unordered_set<Point_3> unique_points;
    for (int i = 0; i < n_points(); i++)
        unique_points.insert(m_pointset.point(i));
    m_pointset.clear();
    for (const auto& p : unique_points) 
    {
        m_pointset.insert(p);
    }
}

AlphaShapeData PointCloud::alpha_shape_2d(double alpha)
{
    std::vector<Point_2> points;
    std::vector<Segment_2> segments;
    //projection first
    for (int i = 0; i < n_points(); i++)
    {
        Point_2 point(m_pointset.point(i).x(), m_pointset.point(i).y());
        points.push_back(point);
    }
    Alpha_shape_2 A(points.begin(), points.end(), FT(alpha), Alpha_shape_2::GENERAL);
    FT optimal_alpha = *A.find_optimal_alpha(1);
    //print info
    spdlog::info("Alpha Shape computed, Optimal alpha: {}", *A.find_optimal_alpha(1));
    alpha_edges(A, std::back_inserter(segments));
    spdlog::info("{} alpha shape edges", segments.size());
    return AlphaShapeData(segments, optimal_alpha);
}
std::vector<Segment_2> PointCloud::connect_segments(std::vector<Segment_2>& segments)
{
    std::vector<Segment_2> connected_segments;
    std::unordered_set<Point_2> visited_points;
    connected_segments.push_back(segments[0]);
    //visited_points.insert(segments[0].source());
    visited_points.insert(segments[0].target());
    segments.erase(segments.begin());
    //print info
    spdlog::info("connect segments...");
    while (!segments.empty()) {
        bool found = false;
        for (int i = 0; i < segments.size(); i++) {
            if (visited_points.count(segments[i].source())) {
                connected_segments.push_back(segments[i]);
                visited_points.insert(segments[i].target());
                segments.erase(segments.begin() + i);
                found = true;
                break;
            }
        }
        if (!found) {
            spdlog::warn("Some segments are deleted");
            break;
        }
    }
    spdlog::info("segments connected!");
    return connected_segments;
}

std::vector<Primitive> PointCloud::improved_region_grow(double eta, int min_support)
{
    //m_pointset : input points with optimized normal
    //min_support : minimum num of points reuired to form a plane
    std::vector<Primitive> primitives; //return value
    
    Eigen::MatrixXd optimized_normals = GetNormals();
    EstimateNormal(30); // get original normal
    std::vector<Vector_3> optimizedN, originalN;
    for (int i = 0; i < n_points(); i++)
    {
        optimizedN.push_back(Vector_3(optimized_normals(i, 0), optimized_normals(i, 1), optimized_normals(i, 2)));
        originalN.push_back(m_pointset.normal(i));
    }
    // region grow
    std::unordered_set<int> remainPoints;
    int Npoints = n_points();
    for (int i = 0; i < Npoints; i++)
        remainPoints.insert(i);
    buildKDTree(); // build kdtree for search neighbors
    spdlog::info("start improved region growing...");
    Timer timer;
    timer.start();
    while (remainPoints.size() > min_support)
    {
        Point_vector points;
        std::vector<Point_3> fit_points;
        int start_id = *(remainPoints.begin());
        points.push_back(boost::make_tuple(m_pointset.point(start_id), originalN[start_id], optimizedN[start_id], start_id));
        fit_points.push_back(m_pointset.point(start_id));
        remainPoints.erase(start_id);
        int iter = 0;
        while (iter < points.size())
        {
            PNNI point = points[iter];
            std::vector<int> neighbors = K_Neighbors(point.get<3>(), 10);
            for (int n_id : neighbors)
            {
                if (remainPoints.find(n_id) == remainPoints.end())
                    continue;
                if (optimizedN[n_id] == point.get<2>() && angle(originalN[n_id], point.get<1>(), true) < DegToRad(eta))
                {
                    points.push_back(boost::make_tuple(m_pointset.point(n_id), originalN[n_id], optimizedN[n_id], n_id));
                    fit_points.push_back(m_pointset.point(n_id));
                    remainPoints.erase(n_id);
                }
            }
            iter++;
        }
        if (points.size() > min_support)
        {
            //fit a plane for the points
            Plane_3 plane;
            CGAL::linear_least_squares_fitting_3(fit_points.begin(), fit_points.end(), plane, CGAL::Dimension_tag<0>());
            primitives.push_back(Primitive(points, plane));
        }
    }
    spdlog::info("Done. Time : {} s.", timer.time());
    spdlog::info("Extract {} planar primitives.", primitives.size());

    //save primitives to m_primitives for backup
    m_primitives.clear();
    for (auto primitive : primitives)
        m_primitives.push_back(primitive);
    
    return primitives;
}

std::vector<Vector_3> PointCloud::reference_direction()
{
    // input pointset with L0 optimized normals
    std::vector<Vector_3> ref_dirs; // return value
    std::vector<Vector_3> optimized_normals, original_normals;
    for (int i = 0; i < n_points(); i++)
    {
        optimized_normals.push_back(m_pointset.normal(i));
        if (!exsist(m_pointset.normal(i), ref_dirs))
        {
            ref_dirs.push_back(m_pointset.normal(i));
        }
    }
    EstimateNormal(30); // get original normals
    for (int i = 0; i < n_points(); i++)
    {
        original_normals.push_back(m_pointset.normal(i));
        m_pointset.normal(i) = optimized_normals[i];
    }
    //comp
    //compute error
    std::vector<int> index_arr;
    std::vector<double> error_arr;
    for (int i = 0; i < n_points(); i++)
    {
        Vector_3 v = optimized_normals[i];
        for(int j = 0; j < ref_dirs.size(); j++)
            if (v == ref_dirs[j])
            {
                index_arr.push_back(j);
                break;
            }
        Vector_3 e_vect = optimized_normals[i] - original_normals[i];
        error_arr.push_back(e_vect.squared_length());
    }
    std::array<double, 3> error = { 0.0, 0.0, 0.0 };
    std::array<double, 3> num = {0, 0, 0};
    for (int i = 0; i < n_points(); i++)
    {
        error[index_arr[i]] += error_arr[i];
        num[index_arr[i]] += 1;
    }
    spdlog::info("dir 1 : ({:.5f},{:.5f},{:.5f}) err : {:.5f}", ref_dirs[0].x(), ref_dirs[0].y(), ref_dirs[0].z(), error[0] / num[0]);
    spdlog::info("dir 2 : ({:.5f},{:.5f},{:.5f}) err : {:.5f}", ref_dirs[1].x(), ref_dirs[1].y(), ref_dirs[1].z(), error[1] / num[1]);
    spdlog::info("dir 3 : ({:.5f},{:.5f},{:.5f}) err : {:.5f}", ref_dirs[2].x(), ref_dirs[2].y(), ref_dirs[2].z(), error[2] / num[2]);
    //sort
    std::vector<boost::tuple<Vector_3, double>> sort_err;
    for (int i = 0; i < 3; i++)
        sort_err.push_back(boost::make_tuple(ref_dirs[i], error[i] / num[i]));
    std::sort(sort_err.begin(), sort_err.end(), [](const auto& tuple1, const auto& tuple2) {
        return boost::get<1>(tuple1) < boost::get<1>(tuple2);
        });
    ref_dirs.clear();
    for (int i = 0; i < 3; i++)
        ref_dirs.push_back(sort_err[i].get<0>());

    return ref_dirs;
}

std::vector<Vector_3> PointCloud::orthogonal_direction(const std::vector<Vector_3>& ref_dirs)
{
    std::vector<Vector_3> ortho_dirs; //return value
    Vector_3 r1 = ref_dirs[0];
    Vector_3 r2 = ref_dirs[1];
    Vector_3 r3 = ref_dirs[2];
    Vector_3 r1_ = r1;
    Vector_3 r2_ = r2 - dot_vector(r2, r1_) / dot_vector(r1_, r1_) * r1_;
    Vector_3 r3_ = r3 - dot_vector(r3, r1_) / dot_vector(r1_, r1_) * r1_ - dot_vector(r3, r2_) / dot_vector(r2_, r2_) * r2_;
    ortho_dirs.push_back(r1_);
    ortho_dirs.push_back(r2_);
    ortho_dirs.push_back(r3_);
    spdlog::info("ortho dir 1 : ({:.5f},{:.5f},{:.5f})", r1_.x(), r1_.y(), r1_.z());
    spdlog::info("ortho dir 2 : ({:.5f},{:.5f},{:.5f})", r2_.x(), r2_.y(), r2_.z());
    spdlog::info("ortho dir 3 : ({:.5f},{:.5f},{:.5f})", r3_.x(), r3_.y(), r3_.z());

    return ortho_dirs;
}

void PointCloud::set_parallel(std::vector<Primitive>& primitives, std::vector<Vector_3>& ref_dirs, std::vector<Vector_3>& ortho_dirs, double tau)
{
    int count = 0;
    for (Primitive& primitive : primitives)
    {
        for (int i = 0; i < ref_dirs.size(); i++)
        {
            if (angle(ref_dirs[i], primitive.plane.orthogonal_vector(), true) < DegToRad(tau))
            {
                primitive.set_target_normal(ortho_dirs[i]);
                count++;
                break;
            }
        }
    }
    spdlog::info("{} primitives are set parallel and perpendicular const.", count);
}

void PointCloud::set_symmetry(std::vector<Primitive>& primitives, std::vector<Vector_3>& ortho_dirs, double epsilon)
{
    // reqirements : parallel and perpendicular const. are set already
    std::vector<boost::tuple<Primitive, int>> primitives_set; // int : the idx in the input primitives
    for (int i = 0; i < primitives.size(); i++)
        if (primitives[i].target_normal == Vector_3(0.0, 0.0, 0.0))
            primitives_set.push_back(boost::make_tuple(primitives[i], i));
    spdlog::info("search symmetry in {} primitives", primitives_set.size());
    std::unordered_set<int> remainPrimitives;
    for (int i = 0; i < primitives_set.size(); i++)
        remainPrimitives.insert(i);
    Vector_3 or1 = ortho_dirs[0];
    Vector_3 or2 = ortho_dirs[1];
    Vector_3 or3 = ortho_dirs[2];
    Eigen::Matrix3d C;
    C << or1[0], or2[0], or3[0],
        or1[1], or2[1], or3[1],
        or1[2], or2[2], or3[2];
    //transform the normal of the plane in the primitives_set
    spdlog::info("transform normal of the plane");
    for (int i = 0; i < primitives_set.size(); i++)
    {
        Vector_3 normal = primitives_set[i].get<0>().plane.orthogonal_vector();
        Eigen::Vector3d n;
        n << normal[0], normal[1], normal[2];
        Eigen::Matrix3d C_ = C.inverse();
        Eigen::Vector3d tn = C_ * n;
        primitives_set[i].get<0>().set_transformed_normal(Vector_3(tn(0), tn(1), tn(2)));
    }
    //detect symmetry in three directions
    //r1*-o-r3*
    spdlog::info("search symmetry about r1Or3");
    for (int i = 0; i < primitives_set.size(); i++)
    {
        if (remainPrimitives.find(i) == remainPrimitives.end())
            continue;
        std::vector<Vector_3> normal_set;
        std::vector<int> idx_set;
        Vector_3 ni = primitives_set[i].get<0>().transformed_normal;
        normal_set.push_back(ni);
        idx_set.push_back(i);
        for (int j = 0; j < primitives_set.size(); j++)
        {
            if (j == i || remainPrimitives.find(j) == remainPrimitives.end())
                continue;
            Vector_3 nj = primitives_set[j].get<0>().transformed_normal;
            bool ispara = abs(ni[0] - nj[0]) < epsilon && abs(ni[1] - nj[1]) < epsilon && abs(ni[2] - nj[2]) < epsilon;
            bool issymm = abs(ni[0] - nj[0]) < epsilon && (ni[1] + nj[1]) < epsilon && abs(ni[2] - nj[2]) < epsilon;
            if (ispara || issymm)
            {
                normal_set.push_back(nj);
                idx_set.push_back(j);
            }
        }
        //avarage normal update
        //x,z : avg, y : max and min
        if (normal_set.size() > 1)
        {
            double x_avg = 0.0;
            double z_avg = 0.0;
            double y_max = std::numeric_limits<double>::min();
            double y_min = std::numeric_limits<double>::max();
            for (int i = 0; i < normal_set.size(); i++)
            {
                x_avg += normal_set[i][0];
                z_avg += normal_set[i][2];
                y_max = std::max(y_max, normal_set[i][1]);
                y_min = std::min(y_min, normal_set[i][1]);
                remainPrimitives.erase(idx_set[i]);
            }
            x_avg /= normal_set.size();
            z_avg /= normal_set.size();
            double y_abs = (y_max - y_min) / 2;
            //update
            for (int i = 0; i < idx_set.size(); i++)
            {
                double y_sign = 1.0;
                if (primitives_set[idx_set[i]].get<0>().transformed_normal[1] < 0)
                    y_sign = -1.0;
                primitives_set[idx_set[i]].get<0>().set_transformed_normal(Vector_3(x_avg, y_sign * y_abs, z_avg));
                primitives_set[idx_set[i]].get<0>().set_target_normal(Vector_3(x_avg, y_sign * y_abs, z_avg));
            }
        }
    }
    int remain = remainPrimitives.size();
    spdlog::info("{} primitves are set r1Or3 symm", primitives_set.size() - remainPrimitives.size());
    //r2*-o-r3*
    spdlog::info("search symmetry about r2Or3");
    for (int i = 0; i < primitives_set.size(); i++)
    {
        if (remainPrimitives.find(i) == remainPrimitives.end())
            continue;
        std::vector<Vector_3> normal_set;
        std::vector<int> idx_set;
        Vector_3 ni = primitives_set[i].get<0>().transformed_normal;
        normal_set.push_back(ni);
        idx_set.push_back(i);
        for (int j = 0; j < primitives_set.size(); j++)
        {
            if (j == i || remainPrimitives.find(j) == remainPrimitives.end())
                continue;
            Vector_3 nj = primitives_set[j].get<0>().transformed_normal;
            bool ispara = abs(ni[0] - nj[0]) < epsilon && abs(ni[1] - nj[1]) < epsilon && abs(ni[2] - nj[2]) < epsilon;
            bool issymm = (ni[0] + nj[0]) < epsilon && abs(ni[1] - nj[1]) < epsilon && abs(ni[2] - nj[2]) < epsilon;
            if (ispara || issymm)
            {
                normal_set.push_back(nj);
                idx_set.push_back(j);
            }
        }
        //avarage normal update
        //y,z : avg, x : max and min
        if (normal_set.size() > 1)
        {
            double y_avg = 0.0;
            double z_avg = 0.0;
            double x_max = std::numeric_limits<double>::min();
            double x_min = std::numeric_limits<double>::max();
            for (int i = 0; i < normal_set.size(); i++)
            {
                y_avg += normal_set[i][1];
                z_avg += normal_set[i][2];
                x_max = std::max(x_max, normal_set[i][0]);
                x_min = std::min(x_min, normal_set[i][0]);
                remainPrimitives.erase(idx_set[i]);
            }
            y_avg /= normal_set.size();
            z_avg /= normal_set.size();
            double x_abs = (x_max - x_min) / 2;
            //update
            for (int i = 0; i < idx_set.size(); i++)
            {
                double x_sign = 1.0;
                if (primitives_set[idx_set[i]].get<0>().transformed_normal[0] < 0)
                    x_sign = -1.0;
                primitives_set[idx_set[i]].get<0>().set_transformed_normal(Vector_3(x_sign* x_abs, y_avg, z_avg));
                primitives_set[idx_set[i]].get<0>().set_target_normal(Vector_3(x_sign * x_abs, y_avg, z_avg));
            }
        }
    }
    spdlog::info("{} primitves are set r2Or3 symm", remain - remainPrimitives.size());
    remain = remainPrimitives.size();
    //r2*-o-r3*
    spdlog::info("search symmetry about r1Or2");
    for (int i = 0; i < primitives_set.size(); i++)
    {
        if (remainPrimitives.find(i) == remainPrimitives.end())
            continue;
        std::vector<Vector_3> normal_set;
        std::vector<int> idx_set;
        Vector_3 ni = primitives_set[i].get<0>().transformed_normal;
        normal_set.push_back(ni);
        idx_set.push_back(i);
        for (int j = 0; j < primitives_set.size(); j++)
        {
            if (j == i || remainPrimitives.find(j) == remainPrimitives.end())
                continue;
            Vector_3 nj = primitives_set[j].get<0>().transformed_normal;
            bool ispara = abs(ni[0] - nj[0]) < epsilon && abs(ni[1] - nj[1]) < epsilon && abs(ni[2] - nj[2]) < epsilon;
            bool issymm = abs(ni[0] - nj[0]) < epsilon && abs(ni[1] - nj[1]) < epsilon && (ni[2] + nj[2]) < epsilon;
            if (ispara || issymm)
            {
                normal_set.push_back(nj);
                idx_set.push_back(j);
            }
        }
        //avarage normal update
        //x,y : avg, z : max and min
        if (normal_set.size() > 1)
        {
            double x_avg = 0.0;
            double y_avg = 0.0;
            double z_max = std::numeric_limits<double>::min();
            double z_min = std::numeric_limits<double>::max();
            for (int i = 0; i < normal_set.size(); i++)
            {
                x_avg += normal_set[i][0];
                y_avg += normal_set[i][1];
                z_max = std::max(z_max, normal_set[i][2]);
                z_min = std::min(z_min, normal_set[i][2]);
                remainPrimitives.erase(idx_set[i]);
            }
            x_avg /= normal_set.size();
            y_avg /= normal_set.size();
            double z_abs = (z_max - z_min) / 2;
            //update
            for (int i = 0; i < idx_set.size(); i++)
            {
                double z_sign = 1.0;
                if (primitives_set[idx_set[i]].get<0>().transformed_normal[2] < 0)
                    z_sign = -1.0;
                primitives_set[idx_set[i]].get<0>().set_transformed_normal(Vector_3(x_avg, y_avg, z_sign* z_abs));
                primitives_set[idx_set[i]].get<0>().set_target_normal(Vector_3(x_avg, y_avg, z_sign* z_abs));
            }
        }
    }
    spdlog::info("{} primitves are set r1Or2 symm", remain - remainPrimitives.size());
    //transform back
    spdlog::info("transform normal of the plane back");
    for (int i = 0; i < primitives_set.size(); i++)
    {
        Vector_3 normal = primitives_set[i].get<0>().transformed_normal;
        Eigen::Vector3d n;
        n << normal[0], normal[1], normal[2];
        Eigen::Vector3d tn = C * n;
        primitives_set[i].get<0>().set_transformed_normal(Vector_3(tn(0), tn(1), tn(2)));
        if (primitives_set[i].get<0>().target_normal != Vector_3(0.0, 0.0, 0.0))
            primitives_set[i].get<0>().set_target_normal(primitives_set[i].get<0>().transformed_normal);
    }
    //change the value in the input primitives
    for (int i = 0; i < primitives_set.size(); i++)
    {
        primitives[primitives_set[i].get<1>()].set_target_normal(primitives_set[i].get<0>().target_normal);
    }
    spdlog::info("{} primitives are set symmetry", primitives_set.size() - remainPrimitives.size());
}

void PointCloud::set_coplanar(std::vector<Primitive>& primitives, double delta)
{
    //requirements : set_parallel and set_symmetry already
    std::vector<Primitive> mergedPrimitives;
    std::unordered_set<int> remainPrimitives;
    for (int i = 0; i < primitives.size(); i++)
        remainPrimitives.insert(i);
    for (int i = 0; i < primitives.size(); i++)
    {
        if (remainPrimitives.find(i) == remainPrimitives.end() || primitives[i].target_normal == Vector_3(0.0, 0.0, 0.0))
            continue;
        std::vector<Primitive> merge_prim;
        std::vector<int> merge_id;
        merge_prim.push_back(primitives[i]);
        merge_id.push_back(i);
        for (int j = 0; j < primitives.size(); j++)
        {
            if (j == i || remainPrimitives.find(j) == remainPrimitives.end() || primitives[j].target_normal == Vector_3(0.0, 0.0, 0.0))
                continue;
            bool ispara = primitives[i].target_normal == primitives[j].target_normal;
            bool isnear = abs(primitives[i].plane.d() - primitives[j].plane.d()) < delta;
            if (ispara && isnear)
            {
                merge_prim.push_back(primitives[j]);
                merge_id.push_back(j);
            }
        }
        if (merge_id.size() > 1)
        {
            Primitive mergePrim = merge_primives(merge_prim);
            mergedPrimitives.push_back(mergePrim);
            for (int i = 0; i < merge_id.size(); i++)
                remainPrimitives.erase(merge_id[i]);
        }
    }
    //add remainPrimitives
    for (int i : remainPrimitives)
        mergedPrimitives.push_back(primitives[i]);
    spdlog::info("{} primitives merged, remain {} primitives", primitives.size() - remainPrimitives.size(), mergedPrimitives.size());
    //change the input primitives
    primitives.clear();
    for (Primitive prim : mergedPrimitives)
        primitives.push_back(prim);
}

Primitive PointCloud::merge_primives(const std::vector<Primitive>& primitives)
{
    Point_vector mergepoints;
    std::vector<Point_3> fit_points;
    for (Primitive prim : primitives)
        for (PNNI pnni : prim.points)
        {
            mergepoints.push_back(pnni);
            fit_points.push_back(pnni.get<0>());
        }
    Plane_3 plane;
    CGAL::linear_least_squares_fitting_3(fit_points.begin(), fit_points.end(), plane, CGAL::Dimension_tag<0>());
    Primitive mergedPrimitive(mergepoints, plane);
    mergedPrimitive.set_target_normal(primitives[0].target_normal);
    
    return mergedPrimitive;
}

double energyfunc(const std::vector<double>& x, std::vector<double>& grad, void* my_func_data)
{
    static int count = 0;
    spdlog::info("iteration {}", ++count);
    // x : (alpha, beta, zeta, d1...dn), my_func_data : pointer to the primitives
    std::vector<Primitive>* primitives = static_cast<std::vector<Primitive>*>(my_func_data);
    std::array<Eigen::Matrix3d, 6> rmatrix = getRotationMat(x[0], x[1], x[2]);
    Eigen::Matrix3d R1 = rmatrix[0], R2 = rmatrix[1], R3 = rmatrix[2];
    Eigen::Matrix3d dR1 = rmatrix[3], dR2 = rmatrix[4], dR3 = rmatrix[5];

    if (!grad.empty()) {
        for (int i = 0; i < grad.size(); i++)
            grad[i] = 0.0;
        for (int i = 0; i < primitives->size(); i++)
        {
            Vector_3 tn = (*primitives)[i].target_normal;
            Eigen::Vector3d vtn, vtn0, vtn1, vtn2;
            vtn << tn[0], tn[1], tn[2];
            vtn0 = vtn; vtn1 = vtn; vtn2 = vtn;
            vtn = R1 * R2 * R3 * vtn;
            vtn0= dR1 * R2 * R3 * vtn;
            vtn1 = R1 * dR2 * R3 * vtn;
            vtn2 = R1 * R2 * dR3 * vtn;
            vtn = vtn.normalized();
            vtn0 = vtn.normalized();
            vtn1 = vtn.normalized();
            vtn2 = vtn.normalized();
            Eigen::Vector4d vf, vf0, vf1, vf2;
            vf << vtn(0), vtn(1), vtn(2), x[i + 3];
            vf0 << vtn0(0), vtn0(1), vtn0(2), 0;
            vf1 << vtn1(0), vtn1(1), vtn1(2), 0;
            vf2 << vtn2(0), vtn2(1), vtn2(2), 0;
            for (int j = 0; j < (*primitives)[i].points.size(); j++)
            {
                Point_3 p = (*primitives)[i].points[j].get<0>();
                Eigen::Vector4d vj;
                vj << p[0], p[1], p[2], 1;
                grad[0] = grad[0] + 2 * vf.dot(vj) * vf0.dot(vj);
                grad[1] = grad[1] + 2 * vf.dot(vj) * vf1.dot(vj);
                grad[2] = grad[2] + 2 * vf.dot(vj) * vf2.dot(vj);
                grad[i] = grad[i] + 2 * vf.dot(vj);
            }
        }
    }
    double result = 0.0;
    for (int i = 0; i < primitives->size(); i++)
    {
        //compute the points' distance to plane
        Vector_3 tn = (*primitives)[i].target_normal;
        Eigen::Vector3d vtn;
        vtn << tn[0], tn[1], tn[2];
        vtn = R1 * R2 * R3 * vtn;
        vtn = vtn.normalized();
        Plane_3 plane(vtn(0), vtn(1), vtn(2), x[i + 3]);
        for (int j = 0; j < (*primitives)[i].points.size(); j++)
        {
            Point_3 p = (*primitives)[i].points[j].get<0>();
            double squaredist = CGAL::squared_distance(p, plane);
            result += squaredist;
        }
    }
    return result;
}

void PointCloud::primitive_optimize(std::vector<Primitive>& primitives, std::array<double, 3>& angles)
{
    // rotation angles : alpha, beta, zeta
    std::vector<Primitive> zprim, optprim;
    for (int i = 0; i < primitives.size(); i++)
    {
        if (primitives[i].target_normal == Vector_3(0.0, 0.0, 0.0))
            zprim.push_back(primitives[i]);
        else
            optprim.push_back(primitives[i]);
    }
    //optimization
    Timer timer;
    timer.start();
    spdlog::info("start optimization");
    nlopt::opt opt(nlopt::LN_COBYLA, optprim.size() + 3);
    opt.set_min_objective(energyfunc, &optprim);
    opt.set_xtol_rel(1e-4);
    //opt.set_maxeval(1000);
    std::vector<double> x = { 0.0,0.0,0.0 };
    for (int i = 0; i < optprim.size(); i++)
        x.push_back(optprim[i].plane.d());
    double minf;
    try {
        nlopt::result result = opt.optimize(x, minf);
        spdlog::info("converge. Time {:.3f} s, alpha {:3f}, beta {:.3f}, zeta {:.3f}",
            timer.time(), RadToDeg(x[0]), RadToDeg(x[1]), RadToDeg(x[2]));
    }
    catch (std::exception& e) {
        spdlog::error("optimization failed!");
    }
    //update the primitives
    primitives.clear();
    for (auto prim : zprim)
        primitives.push_back(prim);
    for (int i = 0; i < optprim.size(); i++)
    {
        Vector_3 tn = optprim[i].target_normal;
        Eigen::Vector3d vtn;
        vtn << tn[0], tn[1], tn[2];
        vtn = vtn.normalized();
        std::array<Eigen::Matrix3d, 6> rmatrix = getRotationMat(x[0], x[1], x[2]);
        Eigen::Matrix3d R1 = rmatrix[0], R2 = rmatrix[1], R3 = rmatrix[2];
        vtn = R1 * R2 * R3 * vtn;
        Plane_3 plane(vtn(0), vtn(1), vtn(2), x[i + 3]);
        Primitive prim(optprim[i].points, plane);
        prim.set_target_normal(optprim[i].target_normal);
        primitives.push_back(prim);
    }
}

void PointCloud::ProjectPrim(std::vector<Primitive>& primitives)
{
    for (int i = 0; i < primitives.size(); i++)
    {
        for (int j = 0; j < primitives[i].points.size(); j++)
        {
            Point_3 p = primitives[i].points[j].get<0>();
            Point_3 q = primitives[i].plane.projection(p);
            primitives[i].points[j].get<0>() = q;
        }
    }
}