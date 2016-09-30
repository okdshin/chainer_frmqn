#include <iostream>
#include <tuple>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define EIGEN_NO_DEBUG
//#define EIGEN_DONT_PARALLELIZE
#define EIGEN_MPL2_ONLY
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Dense>

uint8_t clamp(float v, float low, float high) {
	return static_cast<uint8_t>(std::max(0.0f, std::min(v, high)));
}

std::tuple<bool, float> intersect_triangle(
		std::vector<float> const& origin,
		Eigen::Vector3f const& ray,
		Eigen::Vector3f const& e1,
		Eigen::Vector3f const& e2,
		Eigen::Vector3f const& v0) {
	using namespace Eigen;
	auto A = MatrixXf(3, 3);
	A << e1, e2, -ray;
	auto x = static_cast<Vector3f>(A.householderQr().solve(Vector3f(origin.data())-v0));
	if(0 <= x[0] && 0 <= x[1] && x[0]+x[1] <= 1.0 && 0 < x[2]) {
		return std::make_tuple(true, x[2]);
	}
	return std::make_tuple(false, 0.0);
}

std::vector<std::vector<std::vector<uint8_t>>> raytrace(int image_width, int image_height,
		std::vector<float> const& origin, float f, float yaw, float pitch,
		std::vector<std::vector<std::vector<float>>> const& triangles,
		std::vector<std::vector<uint8_t>> const& colors) {
	using namespace Eigen;
	auto image = std::vector<std::vector<std::vector<uint8_t>>>(
		image_height,
		std::vector<std::vector<uint8_t>>(image_width, {0, 0, 0})
	);
	for(auto y = 0; y < image_height; ++y) {
		for(auto x = 0; x < image_width; ++x) {
			auto dir = Vector3f::UnitY();
			auto roty = Quaternionf(Eigen::AngleAxisf(M_PI*yaw/180.f, Vector3f::UnitZ()));
			auto pitch_axis = static_cast<Vector3f>(roty*Vector3f::UnitX());
			auto rotp = Quaternionf(AngleAxisf(M_PI*pitch/180.f, pitch_axis));
			auto pixel = static_cast<Vector3f>(Vector3f(image_width-x, 0, image_height-y)-Vector3f(image_width/2.0, 0, image_height/2.0));
			auto ray = static_cast<Vector3f>(rotp * roty * (pixel + f*dir));
			ray.normalize();
			//std::cout << "pixel" << pixel << std::endl;
			//std::cout << f << " " << dir << std::endl;
			//std::cout << ray << "\n" << std::endl;
			auto min_distance = std::numeric_limits<float>::max();
			auto pixcel_color = std::vector<uint8_t>({0, 0, 0});
			for(auto i=0u; i < triangles.size(); ++i) {
				assert(triangles[i].size() == 3);
				auto v0 = Vector3f(triangles[i][0].data());
				auto v1 = Vector3f(triangles[i][1].data());
				auto v2 = Vector3f(triangles[i][2].data());
				auto e1 = static_cast<Vector3f>(v1-v0);
				auto e2 = static_cast<Vector3f>(v2-v0);
				bool intersected; float distance; std::tie(intersected, distance)
					= intersect_triangle(origin, ray, e1, e2, v0);
				if(intersected && 0 < distance && distance < min_distance) {
					// TODO intensity
					auto intensity = -ray.dot(e1.cross(e2).normalized());
					//std::cout << intensity << std::endl;
					//std::cout << "colors" << int(colors[i][0]) << " " << int(colors[i][1]) << " " << int(colors[i][2]) << std::endl;
					/*
					auto color = static_cast<Vector3f>(
							Vector3f(colors[i][0], colors[i][1], colors[i][2]));
					*/
					auto color = static_cast<Vector3f>(
							Vector3f(colors[i][0], colors[i][1], colors[i][2])
							+(intensity-1.0)*Vector3f(50,50,50));
					min_distance = distance;
					pixcel_color[0] = clamp(color[0], 0, 255);
					pixcel_color[1] = clamp(color[1], 0, 255);
					pixcel_color[2] = clamp(color[2], 0, 255);
					/*
					assert(0 <= colors[i][0] && colors[i][0] < 256);
					assert(0 <= colors[i][1] && colors[i][1] < 256);
					assert(0 <= colors[i][2] && colors[i][2] < 256);
					pixcel_color[0] = colors[i][0];
					pixcel_color[1] = colors[i][1];
					pixcel_color[2] = colors[i][2];
					*/
					//std::cout << int(pixcel_color[0]) << std::endl;
				}
			}
			image[y][x] = pixcel_color;
			//std::cout << "image" << int(image[y][x][0]) << " " 
				//<< int(image[y][x][1]) << " " << int(image[y][x][2])<< std::flush;
		}
	}
	return image;
}

std::vector<std::vector<std::vector<uint8_t>>> raytrace2(int image_width, int image_height,
		std::vector<float> const& origin, float f, float yaw, float pitch,
		std::vector<std::vector<std::vector<float>>> const& triangles,
		std::vector<std::vector<uint8_t>> const& colors) {
	using namespace Eigen;
	auto image = std::vector<std::vector<std::vector<uint8_t>>>(
		image_height,
		std::vector<std::vector<uint8_t>>(image_width, {0, 0, 0})
	);
	for(auto y = 0; y < image_height; ++y) {
		for(auto x = 0; x < image_width; ++x) {
			auto dir = Vector3f::UnitY();
			auto roty = Quaternionf(Eigen::AngleAxisf(M_PI*yaw/180.f, Vector3f::UnitZ()));
			auto pitch_axis = static_cast<Vector3f>(roty*Vector3f::UnitX());
			auto rotp = Quaternionf(AngleAxisf(M_PI*pitch/180.f, pitch_axis));
			auto pixel = static_cast<Vector3f>(Vector3f(image_width-x, 0, image_height-y)-Vector3f(image_width/2.0, 0, image_height/2.0));
			auto ray = static_cast<Vector3f>(rotp * roty * (pixel + f*dir));
			ray.normalize();
			auto min_distance = std::numeric_limits<float>::max();
			auto pixcel_color = std::vector<uint8_t>({0, 0, 0});

			std::vector<Vector3f> e1_list(triangles.size());
			std::vector<Vector3f> e2_list(triangles.size());

			std::vector<Vector3f> v0_list(triangles.size());
			std::vector<Vector3f> normal_list(triangles.size());
			std::vector<Vector3f> binormal_u_list(triangles.size());
			std::vector<Vector3f> binormal_v_list(triangles.size());
			for(auto i=0u; i < triangles.size(); ++i) {
				v0_list[i] = Vector3f(triangles[i][0].data());
				auto v1 = Vector3f(triangles[i][1].data());
				auto v2 = Vector3f(triangles[i][2].data());
				e1_list[i] = static_cast<Vector3f>(v1-v0_list[i]);
				e2_list[i] = static_cast<Vector3f>(v2-v0_list[i]);
				normal_list[i] = static_cast<Vector3f>(e1_list[i].cross(e2_list[i]));
				auto nx = static_cast<Vector3f>(e2_list[i].cross(normal_list[i]));
				auto ny = static_cast<Vector3f>(e1_list[i].cross(normal_list[i]));
				binormal_u_list[i] = nx/e1_list[i].dot(nx);
				binormal_v_list[i] = ny/e2_list[i].dot(ny);
			}
			for(auto i=0u; i < triangles.size(); ++i) {
				assert(triangles[i].size() == 3);
				auto origin_prime = static_cast<Vector3f>(Vector3f(origin.data())-v0_list[i]);
				auto nd = origin_prime.dot(normal_list[i]);
				auto nv = -ray.dot(normal_list[i]);
				if(nv < 0) {
					continue;
				}
				auto t = nd/nv;
				if(t < 0) {
					continue;
				}
				auto pos = static_cast<Vector3f>(t*ray+origin_prime);
				auto u = pos.dot(binormal_u_list[i]);
				auto v = pos.dot(binormal_v_list[i]);
				bool intersected = (0 < u && 0 < v && u+v < 1.0);
				if(!intersected) {
					continue;
				}
				float distance = t;

				if(distance < min_distance) {
					auto intensity = -ray.dot(e1_list[i].cross(e2_list[i]).normalized());
					auto color = static_cast<Vector3f>(
							Vector3f(colors[i][0], colors[i][1], colors[i][2])
							+(intensity-1.0)*Vector3f(50,50,50));
					min_distance = distance;
					pixcel_color[0] = clamp(color[0], 0, 255);
					pixcel_color[1] = clamp(color[1], 0, 255);
					pixcel_color[2] = clamp(color[2], 0, 255);
				}
			}
			image[y][x] = pixcel_color;
		}
	}
	return image;
}

namespace py = pybind11;

PYBIND11_PLUGIN(raytrace_cpp) {
    py::module m("raytrace_cpp", "raytrace module for RogueGym");

    m.def("raytrace", &raytrace, "A function which make image");
    m.def("raytrace2", &raytrace2, "A function which make image");

    return m.ptr();
}
