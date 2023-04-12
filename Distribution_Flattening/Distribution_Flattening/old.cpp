#include <functional>
#include <iostream>
#include <vector>

#define PI 3.14159265
#define SAMPLE_POINTS 120u
#define RANGE_MIN 0.0
#define RANGE_MAX 10.0

#define LEARNING_RATE 0.1

enum output {
	y,
	dy_dx,
	dy_d_variance,
};

typedef std::function<double(double x, double variance, output output_type)> MathFunction;
typedef std::vector<double> FunctionGraph;
typedef std::vector<double> Gradiant;

MathFunction generate_gauss_function() {
	auto gauss_function = [=](double x, double variance, double mean, output output_type = y) {
		if (output_type == y)
			return 1 / (variance * sqrt(2 * PI)) * exp(-1 / 2.0 * pow((x - mean) / variance, 2));										// gaussian distribution function
		else if (output_type == dy_dx)
			return 1 / (variance * sqrt(2 * PI)) * -1 / variance * (x - mean) / variance * exp(-1 / 2.0 * pow((x - mean) / variance, 2));		// gaussian distribution derivaive w.r.t. x
		else if (output_type == dy_d_variance) {
			double h = 0.001;
			double variance_plus_h = variance + h;
			return (1 / (variance_plus_h * sqrt(2 * PI)) * exp(-1 / 2.0 * pow((x - mean) / variance_plus_h, 2)) - 1 / (variance * sqrt(2 * PI)) * exp(-1 / 2.0 * pow((x - mean) / variance, 2))) / h;	// gaussian distribution derivative w.r.t. variance
		}
	};
	return gauss_function;
}

FunctionGraph sum_functions(const std::vector<std::pair<MathFunction, double>>& functions) {

	FunctionGraph sum;
	sum.reserve(SAMPLE_POINTS);

	double step_size = (RANGE_MAX - RANGE_MIN) / SAMPLE_POINTS;

	for (unsigned int i = 0; i < SAMPLE_POINTS; i++) {
		double value = 0.0;
		for (const auto& pair : functions)
			value += pair.first(RANGE_MIN + i * step_size, pair.second, y);

		sum.push_back(value);
	}

	return sum;
}

double compute_loss(const FunctionGraph& graph) {
	double step_size = (RANGE_MAX - RANGE_MIN) / SAMPLE_POINTS;

	// loss = mean absolute error
	double loss = 0.0;

	for (int i = 0; i < SAMPLE_POINTS - 1; i++) {
		double derivative = (graph[i + 1] - graph[i]) / step_size;
		double derivative_absolute = std::abs(derivative);
		loss += derivative_absolute;
	}
	loss /= SAMPLE_POINTS - 1;

	return loss;
}

Gradiant compute_gradiant(std::vector<std::pair<MathFunction, double>> functions) {
	double h = 0.001;
	Gradiant gradiant;

	for (int i = 0; i < functions.size(); i++) {
		FunctionGraph sum = sum_functions(functions);
		functions[i].second += h;
		FunctionGraph sum_with_h = sum_functions(functions);
		double d_loss_d_variance = (compute_loss(sum_with_h) - compute_loss(sum)) / h;
		gradiant.push_back(d_loss_d_variance);
	}

	return gradiant;
}

void apply_graidnat(std::vector<std::pair<MathFunction, double>>& functions, const Gradiant& gradiant) {
	for (int i = 0; i < functions.size(); i++) {
		functions[i].second += -gradiant[i] * LEARNING_RATE;
	}
}

void display_graph_on_console(const FunctionGraph& graph) {

	double step_size = (RANGE_MAX - RANGE_MIN) / SAMPLE_POINTS;

	double max = graph[0];
	for (double value : graph)
		if (value > max)
			max = value;

	for (int i = std::ceil(max * 100); i >= 0; i--) {
		for (double value : graph) {
			if (value * 100 > i)
				std::cout << "# ";
			else
				std::cout << "  ";
		}
		std::cout << "\n";
	}
	for (int i = 0; i < graph.size(); i++)
		std::cout << (int)(step_size * i) << " ";
	std::cout << "\n";


}

int main() {
	std::vector<double> current_variances;
	auto gauss1 = generate_gauss_function(1);
	auto gauss2 = generate_gauss_function(3);
	auto gauss3 = generate_gauss_function(4);
	auto gauss4 = generate_gauss_function(6);
	auto gauss5 = generate_gauss_function(12);


	auto functions = std::vector<std::pair<MathFunction, double>>{
		std::pair<MathFunction, double>(gauss1, 1.2),
		std::pair<MathFunction, double>(gauss2, 3.0),
		std::pair<MathFunction, double>(gauss3, 5.0),
		std::pair<MathFunction, double>(gauss4, 2.0),
		std::pair<MathFunction, double>(gauss5, 2.0),
	};

	for (int i = 0; i < 10000; i++) {
		Gradiant gradiant = compute_gradiant(functions);
		apply_graidnat(functions, gradiant);

		if (i % 100 == 0)
			display_graph_on_console(sum_functions(functions));

		if (i % 25 == 0) {
			double loss = compute_loss(sum_functions(functions));
			std::cout << loss << std::endl;
		}
	}

	display_graph_on_console(sum_functions(functions));
	double loss_final = compute_loss(sum_functions(functions));
	std::cout << loss_final << std::endl;

	for (auto& function : functions)
		std::cout << function.second << " ";
	std::cout << std::endl;

	std::cin.get();
}