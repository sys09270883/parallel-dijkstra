#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <omp.h>
#include <queue>
#include <stack>
#include <iomanip>

const int INF = 1e9;

class Graph {
private:
	int chunk_size;
	int node_size;
	int source;
	int target;
	int** adj;
	int* dists;
	bool* visited;
	int* trace;

public:
	Graph(const int thr_size, const int node_size, const int source, const int target)
		: node_size(node_size), source(source), target(target) {
		chunk_size = node_size / thr_size;
		adj = new int* [node_size];
		for (int i = 0; i < node_size; i++) {
			adj[i] = new int[node_size];
		}
		dists = new int[node_size];
		visited = new bool[node_size];
		trace = new int[node_size];
		std::fill(dists, dists + node_size, INF);
		std::fill(visited, visited + node_size, false);
		std::fill(trace, trace + node_size, -1);
	}

	~Graph() {
		for (int i = 0; i < node_size; i++) {
			delete[] adj[i];
		}
		delete[] dists, visited, trace;
	}

	void make_graph(const std::string& csv_file) {
		std::ifstream ifs(csv_file);
		if (!ifs.is_open()) {
			std::cout << "File not open." << std::endl;
			exit(1);
		}

		std::string line, tmp;
		getline(ifs, line, '\n');
		int from, to = 0;

		while (getline(ifs, line)) {
			std::stringstream ss(line);
			while (getline(ss, tmp, ',')) {
				if (tmp == "MAX" || ('0' <= tmp[0] && tmp[0] <= '9')) {
					int dist = tmp == "MAX" ? INF : stoi(tmp);
					add_edge(from, to, dist);
					to++;
				}
				else {
					from = stoi(tmp.substr(1, tmp.size()));
					to = 0;
				}
			}
		}
	}

	void add_edge(int from, int to, int dist) {
		adj[from][to] = dist;
	}

	void parallel_dijkstra() {
		visited[source] = true;
		dists[source] = 0;
		int cur_idx = source;

		while (!visited[target]) {
			int min_dist = INF;
			int min_idx = -1;

#pragma omp parallel
			{
				int local_min_dist = INF;
				int local_min_idx = -1;

#pragma omp for nowait schedule(static, chunk_size)
				for (int next_idx = 0; next_idx < node_size; next_idx++) {
					if (visited[next_idx])
						continue;
					register int cur_dist = dists[cur_idx] + adj[cur_idx][next_idx];
					if (dists[next_idx] > cur_dist) {
						dists[next_idx] = cur_dist;
						trace[next_idx] = cur_idx;
					}
					if (local_min_dist > dists[next_idx]) {
						local_min_dist = dists[next_idx];
						local_min_idx = next_idx;
					}
				}

#pragma omp critical
				{
					if (min_dist > local_min_dist) {
						min_dist = local_min_dist;
						min_idx = local_min_idx;
					}
				}
			}

			cur_idx = min_idx;
			visited[cur_idx] = true;
		}
	}

	void print_result(double elapsed_time) {
		std::stack<int> st;
		int tmp = target;
		while (tmp != source) {
			st.push(tmp);
			tmp = trace[tmp];
		}
		st.push(source);

		std::cout << "path: ";
		while (st.size()) {
			std::cout << 'n' << std::setw(4) << std::setfill('0') << st.top() << ' ';
			st.pop();
		}
		std::cout << '\n' << "cost: " << dists[target];
		std::cout << '\n' << "compute time: " << elapsed_time << '\n';
	}
};

int get_node_size(const std::string& csv_file) {
	std::string num = "";
	for (auto c : csv_file) {
		if ('0' <= c && c <= '9')
			num += c;
	}
	return stoi(num);
}

int main(int argc, char** argv) {
	const int thr_size = atoi(argv[1]);
	const int source = atoi(argv[2]);
	const int target = atoi(argv[3]);
	const std::string csv_file = argv[4];
	const int node_size = get_node_size(csv_file);
	double start_time, end_time, elapsed_time;

	omp_set_num_threads(thr_size);

	std::cout << "Before build graph." << '\n';
	Graph graph(thr_size, node_size, source, target);
	graph.make_graph(csv_file);

	start_time = omp_get_wtime();
	graph.parallel_dijkstra();
	end_time = omp_get_wtime();

	elapsed_time = end_time - start_time;
	graph.print_result(elapsed_time);
	return 0;
}