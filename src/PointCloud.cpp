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

static double angle(const Vector_3& v1, const Vector_3& v2)
{
    double dot_product = v1 * v2;
    double v1_length = std::sqrt(v1.squared_length());
    double v2_length = std::sqrt(v2.squared_length());

    double cos_angle = dot_product / (v1_length * v2_length);
    double angle = std::acos(cos_angle);

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
    if(!m_pointset.has_normal_map())
        EstimateNormal();
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
                if (optimizedN[n_id] == point.get<2>() && angle(originalN[n_id], point.get<1>()) < DegToRad(eta))
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
    spdlog::info("dir 1 : ({:03.2f},{:03.2f},{:03.2f}) err : {:03.2f}", ref_dirs[0].x(), ref_dirs[0].y(), ref_dirs[0].z(), error[0] / num[0]);
    spdlog::info("dir 2 : ({:03.2f},{:03.2f},{:03.2f}) err : {:03.2f}", ref_dirs[1].x(), ref_dirs[1].y(), ref_dirs[1].z(), error[1] / num[1]);
    spdlog::info("dir 3 : ({:03.2f},{:03.2f},{:03.2f}) err : {:03.2f}", ref_dirs[2].x(), ref_dirs[2].y(), ref_dirs[2].z(), error[2] / num[2]);
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

void PointCloud::BilateralNormalSmooth(double sigp, double sign, int itertimes)
{
    //bilateral normal smoothing
    /*
    * sigp: gauss sigma parameter of position term
    * sign: gauss sigma parameter of normal term
    * itertimes£º time of iterations
    */
    std::cout << "Bilateral Normal Smoothing. " <<
        "sigp: " << sigp << " sign: " << sign << " itertimes: " << itertimes << std::endl;
    if (!m_pointset.has_normal_map())
        EstimateNormal();
    int n = n_points();
    int ITER = itertimes;
    while (itertimes > 0)
    {
        std::cout << "iter" << ITER - itertimes << std::endl;
        Eigen::MatrixXd N_new = Eigen::MatrixXd::Zero(n, 3);
        //#pragma omp parallel for num_threads(2)
        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            std::vector<int> neighbors = K_Neighbors(i, 10); //k neighbors of point[i]
            for (int j : neighbors)
            {
                //position term
                double sdis = CGAL::squared_distance(Point(i), Point(j));
                double pterm = GaussFunc(sqrt(sdis), sigp);
                //normal term
                double nproduct = CGAL::scalar_product(Normal(i), Normal(j));
                double nterm = GaussFunc(1 - nproduct, 1 - cos(DegToRad(sign)));
                N_new(i, 0) += pterm * nterm * Normal(j).x();
                N_new(i, 1) += pterm * nterm * Normal(j).y();
                N_new(i, 2) += pterm * nterm * Normal(j).z();
                sum += pterm * nterm;
            }
            N_new(i, 0) /= sum;
            N_new(i, 1) /= sum;
            N_new(i, 2) /= sum;
        }
        //update normal with the N_new
        for (int i = 0; i < n; i++)
        {
            Vector_3 v(N_new(i, 0), N_new(i, 1), N_new(i, 2));
            m_pointset.normal(i) = v;
        }

        itertimes--;
    }
    std::cout << "Done" << std::endl;
}