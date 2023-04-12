
#include <iostream>
#include <vector>
#include <string>

#define PI 3.14159265
#define SAMPLE_POINTS 100u
#define RANGE_MIN 0.0
#define RANGE_MAX 10.0

#define LEARNING_RATE 0.1

#define RENDER_HEIGHT_SCALE 20

#define FUNCTION_AMOUNT 400
#define BIG_NUMBER ~(1 << 31)

#define MINIMUM_AMOUNT_OF_FUNCTIONS 30
#define IDEAL_AMOUNT_OF_FUNCTIONS 55

enum output {
	y,
	dy_dx,
	dy_d_variance,
};

typedef std::vector<double> FunctionGraph;

class GaussFunction {
public:

	GaussFunction(double variance = 1.0, double mean = 1.0, double scale = 1.0) :
		variance(variance), mean(mean), scale(scale) {}

	double get_value(double x) const {
		return scale * (1 / (variance * sqrt(2 * PI)) * exp(-1 / 2.0 * pow((x - mean) / variance, 2)));	// gaussian distribution function
	}

	double variance;
	double mean;
	double scale;

	bool operator==(const GaussFunction& other) {
		if (variance != other.variance)
			return false;
		if (mean != other.mean)
			return false;
		if (scale != other.scale)
			return false;
		return true;
	}

	void stay_at_boundry() {
		variance = std::max(variance, 0.1);
		variance = std::min(variance, 8.0);
		mean = std::max(mean, RANGE_MIN - 0.4);
		mean = std::min(mean, RANGE_MAX + 0.4);
		scale = std::max(scale, 0.9);
		scale = std::min(scale, 1.1);
	}

	std::string get_desmos_form() {
		return std::to_string(scale) + "\\ \\cdot\\ \\frac{1}{\\ " + std::to_string(variance) + "\\cdot\\sqrt{2\\pi}\\ \\ }\\cdot\\exp\\left(-\\frac{1}{2.}\\left(\\frac{x - " + std::to_string(mean) + "} {" + std::to_string(variance) + "}\\right)^ { 2 }\\right)";
	}
};

FunctionGraph sum_functions(const std::vector<GaussFunction>& functions) {

	FunctionGraph sum;
	sum.reserve(SAMPLE_POINTS);

	double step_size = (RANGE_MAX - RANGE_MIN) / SAMPLE_POINTS;

	for (unsigned int i = 0; i < SAMPLE_POINTS; i++) {
		double value = 0.0;
		for (const auto& gauss : functions)
			value += gauss.get_value(RANGE_MIN + i * step_size);

		sum.push_back(value);
	}

	return sum;
}

FunctionGraph sum_graphs(const std::vector<FunctionGraph>& graphs) {
	FunctionGraph sum;
	sum.reserve(SAMPLE_POINTS);

	for (unsigned int i = 0; i < SAMPLE_POINTS; i++) {
		double value = 0.0;
		for (const auto& graph : graphs)
			value += graph[i];

		sum.push_back(value);
	}

	return sum;
}

double compute_loss(const FunctionGraph& graph, const std::vector<GaussFunction>& functions) {
	double step_size = (RANGE_MAX - RANGE_MIN) / SAMPLE_POINTS;

	// loss = mean absolute error
	double loss = 0.0;


	double avarage_value = 0;
	for (int i = 0; i < SAMPLE_POINTS - 1; i++) {
		avarage_value += graph[i];
	}
	avarage_value /= SAMPLE_POINTS;

	for (int i = 0; i < SAMPLE_POINTS - 1; i++) {
		double derivative = (graph[i + 1] - graph[i]) / step_size;
		double derivative_absolute = std::abs(derivative);
		double derivative_normalized = derivative_absolute / avarage_value;
		loss += 40 * derivative_normalized;

		double difference_between_avarage = std::abs(graph[i] - avarage_value);
		double difference_between_avarage_normalized = difference_between_avarage / avarage_value;
		loss += 10 * difference_between_avarage_normalized;

		if (graph[i] < 0.4)
			loss += 5.0 * 1.0 / graph[i];
		if (graph[i] > 5.0)
			loss += 5.0;
	}

	for (const GaussFunction& gauss : functions) {
		loss += 5.0 * (std::abs(1 - gauss.scale));
	}

	loss /= SAMPLE_POINTS - 1;

	return loss;
}

void compute_and_apply_gradiant(std::vector<GaussFunction>& functions) {
	double h = 0.001;

	for (int i = 0; i < functions.size(); i++) {
		GaussFunction& gauss = functions[i];

		FunctionGraph sum = sum_functions(functions);
		//gauss.mean += h;
		//gauss.stay_at_boundry();
		//FunctionGraph sum_with_h_mean = sum_functions(functions);
		//double derivative_of_loss_wrt_mean = (compute_loss(sum_with_h_mean, functions) - compute_loss(sum, functions)) / h;
		//gauss.mean -= h;
		//gauss.stay_at_boundry();

		gauss.scale += h;
		gauss.stay_at_boundry();
		FunctionGraph sum_with_h_variance = sum_functions(functions);
		double derivative_of_loss_wrt_scale = (compute_loss(sum_with_h_variance, functions) - compute_loss(sum, functions)) / h;
		gauss.scale -= h;
		gauss.stay_at_boundry();

		//gauss.mean += -derivative_of_loss_wrt_mean * LEARNING_RATE;
		gauss.scale += -derivative_of_loss_wrt_scale * LEARNING_RATE;

		gauss.stay_at_boundry();
	}
}

void display_graph_on_console(const FunctionGraph& graph) {

	double step_size = (RANGE_MAX - RANGE_MIN) / SAMPLE_POINTS;

	double max = graph[0];
	for (double value : graph)
		if (value > max)
			max = value;

	for (int i = std::ceil(max * RENDER_HEIGHT_SCALE); i >= 0; i--) {
		for (double value : graph) {
			if (value * RENDER_HEIGHT_SCALE > i)
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

bool _first_random = true;
double random() {
	if (_first_random)
		std::srand(time(0));
	_first_random = false;

	return (std::rand() % 1000) / 1000.0;
}

void optimize(std::vector<GaussFunction>& functions, unsigned int optimization_iteration = 1000, bool display_function_begin = true, bool display_function_end = true) {
	if (display_function_begin) {
		display_graph_on_console(sum_functions(functions));
		for (auto& function : functions)
			std::cout << "\tvariance: " << function.variance << "\tmean: " << function.mean << "\tscale: " << function.scale << std::endl;
		std::cout << "desmos: " << std::endl;
		bool first_one = true;
		for (auto& function : functions) {
			if (!first_one)
				std::cout << "\\ +\\ ";
			first_one = false;
			std::cout << function.get_desmos_form();
		}
		std::cout << std::endl;
	}

	for (int i = 0; i < optimization_iteration; i++) {
		compute_and_apply_gradiant(functions);

		//if (i % 100 == 0)
		//	display_graph_on_console(sum_functions(functions));

		if (i % 100 == 0) {
			double loss = compute_loss(sum_functions(functions), functions);
			std::cout << loss << std::endl;
			if (loss < 0.018)
				break;
		}
	}

	if (display_function_end) {
		display_graph_on_console(sum_functions(functions));
		for (auto& function : functions) {
			std::cout << "\tvariance: " << function.variance << "\tmean: " << function.mean << "\tscale: " << function.scale << std::endl;
		}
		std::cout << "desmos: " << std::endl;
		bool first_one = true;
		for (auto& function : functions) {
			if (!first_one)
				std::cout << "\\ +\\ ";
			first_one = false;
			std::cout << function.get_desmos_form();
		}
		std::cout << std::endl;
	}
}

int main() {

	std::vector<GaussFunction> all_functions;
	std::vector<GaussFunction> current_functions;

	double best_loss = BIG_NUMBER;
	std::vector<GaussFunction> best_combunation;

	for (int i = 0; i < FUNCTION_AMOUNT; i++) {
		GaussFunction gauss(random() * 0.3, random() * 10, 1.0);
		gauss.stay_at_boundry();
		all_functions.push_back(gauss);
	}

	int iterations_left = IDEAL_AMOUNT_OF_FUNCTIONS;
	while (iterations_left > 0) {

		// find most optimal functions to add
		std::cout << "INSERTING FUNCTIONS" << std::endl;
		for (int episode = 0; episode < 5; episode++) {

			double original_loss = compute_loss(sum_functions(current_functions), current_functions);
			unsigned int min_index = 0;
			double min_loss = BIG_NUMBER;
			for (int i = 0; i < all_functions.size(); i++) {
				auto& gauss = all_functions[i];
				current_functions.push_back(gauss);
				double loss = compute_loss(sum_functions(current_functions), current_functions);
				current_functions.pop_back();
				if (loss < min_loss) {
					min_index = i;
					min_loss = loss;
				}
			}
			//if (original_loss + 0.4 < min_loss)
			//	break;
			current_functions.push_back(all_functions[min_index]);
			all_functions.erase(all_functions.begin() + min_index);

			optimize(current_functions, 0, false, false);
			std::cout << min_loss << std::endl;

			iterations_left--;
			if (current_functions.size() >= MINIMUM_AMOUNT_OF_FUNCTIONS && min_loss < best_loss) {
				best_loss = min_loss;
				best_combunation = current_functions;
			}
		}

		// find most optimal functions to remove
		std::cout << "REMOVING FUNCTIONS" << std::endl;
		for (int episode = 0; episode < 3; episode++) {

			double original_loss = compute_loss(sum_functions(current_functions), current_functions);
			auto min_index = 0;
			double min_loss = BIG_NUMBER;
			for (int i = 0; i < current_functions.size(); i++) {
				auto gauss = current_functions[i];
				current_functions.erase(current_functions.begin() + i);
				double loss = compute_loss(sum_functions(current_functions), current_functions);
				current_functions.insert(current_functions.begin() + i, gauss);
				if (loss < min_loss) {
					min_index = i;
					min_loss = loss;
				}
			}
			//if (original_loss - 0.2 < min_loss)
			//	break;
			all_functions.push_back(current_functions[min_index]);
			current_functions.erase(current_functions.begin() + min_index);

			optimize(current_functions, 0, false, false);
			iterations_left++;
			if (current_functions.size() >= MINIMUM_AMOUNT_OF_FUNCTIONS && min_loss < best_loss) {
				best_loss = min_loss;
				best_combunation = current_functions;
			}
		}
	}

	current_functions = best_combunation;
	optimize(current_functions, 3000);


	std::cin.get();
}