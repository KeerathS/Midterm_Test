#include <math.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stack>
#include <vector>

#include "cartCentering.h"

using namespace std;

// return a double unifomrly sampled in (0,1)
double randDouble(mt19937& rng) {
  return std::uniform_real_distribution<>{0, 1}(rng);
}
// return uniformly sampled 0 or 1
bool randChoice(mt19937& rng) {
  return std::uniform_int_distribution<>{0, 1}(rng);
}
// return a random integer uniformly sampled in (min, max)
int randInt(mt19937& rng, const int& min, const int& max) {
  return std::uniform_int_distribution<>{min, max}(rng);
}

// return true if op is a suported operation, otherwise return false
bool isOp(string op) {
  if (op == "+")
    return true;
  else if (op == "-")
    return true;
  else if (op == "*")
    return true;
  else if (op == "/")
    return true;
  else if (op == ">")
    return true;
  else if (op == "abs")
    return true;
  else
    return false;
}

int arity(string op) {
  if (op == "abs")
    return 1;
  else if (op == "READ_MEM")  // READ_MEM acts as a terminal
    return 0;
  else if (op == "WRITE_MEM") // WRITE_MEM takes one operand
    return 1;
  else
    return 2;
}


typedef string Elem;

class LinkedBinaryTree {
 public:
  struct Node {
    Elem elt;
    string name;
    Node* par;
    Node* left;
    Node* right;
    Node() : elt(), par(NULL), name(""), left(NULL), right(NULL) {}
    int depth() {
      if (par == NULL) return 0;
      return par->depth() + 1;
    }
  };

  class Position {
   private:
    Node* v;

   public:
    Position(Node* _v = NULL) : v(_v) {}
    Elem& operator*() { return v->elt; }
    Position left() const { return Position(v->left); }
    void setLeft(Node* n) { v->left = n; }
    Position right() const { return Position(v->right); }
    void setRight(Node* n) { v->right = n; }
    Position parent() const  // get parent
    {
      return Position(v->par);
    }
    bool isRoot() const  // root of the tree?
    {
      return v->par == NULL;
    }
    bool isExternal() const  // an external node?
    {
      return v->left == NULL && v->right == NULL;
    }
    friend class LinkedBinaryTree;  // give tree access
  };
  typedef vector<Position> PositionList;

 public:
  LinkedBinaryTree() : _root(NULL), score(0), steps(0), generation(0) {}

  // copy constructor
  LinkedBinaryTree(const LinkedBinaryTree& t) {
    _root = copyPreOrder(t.root());
    score = t.getScore();
    steps = t.getSteps();
    generation = t.getGeneration();
  }

  // copy assignment operator
  LinkedBinaryTree& operator=(const LinkedBinaryTree& t) {
    if (this != &t) {
      // if tree already contains data, delete it
      if (_root != NULL) {
        PositionList pl = positions();
        for (auto& p : pl) delete p.v;
      }
      _root = copyPreOrder(t.root());
      score = t.getScore();
      steps = t.getSteps();
      generation = t.getGeneration();
    }
    return *this;
  }

  // destructor
  ~LinkedBinaryTree() {
    if (_root != NULL) {
      PositionList pl = positions();
      for (auto& p : pl) delete p.v;
    }
  }

  int size() const { return size(_root); }
  int size(Node* root) const;
  int depth() const;
  bool empty() const { return size() == 0; };
  Node* root() const { return _root; }
  PositionList positions() const;
  void addRoot() { _root = new Node; }
  void addRoot(Elem e) {
    _root = new Node;
    _root->elt = e;
  }
  void nameRoot(string name) { _root->name = name; }
  void addLeftChild(const Position& p, const Node* n);
  void addLeftChild(const Position& p);
  void addRightChild(const Position& p, const Node* n);
  void addRightChild(const Position& p);
  void printExpression() { printExpression(_root); }
  void printExpression(Node* v);
  double evaluateExpression(double a, double b) {
    return evaluateExpression(Position(_root), a, b);
  };
  double evaluateExpression(const Position& p, double a, double b);
  long getGeneration() const { return generation; }
  void setGeneration(int g) { generation = g; }
  double getScore() const { return score; }
  void setScore(double s) { score = s; }
  double getSteps() const { return steps; }
  void setSteps(double s) { steps = s; }
  void randomExpressionTree(Node* p, const int& maxDepth, mt19937& rng);
  void randomExpressionTree(const int& maxDepth, mt19937& rng) {
    randomExpressionTree(_root, maxDepth, rng);
  }
  void deleteSubtreeMutator(mt19937& rng);
  void addSubtreeMutator(mt19937& rng, const int maxDepth, bool partially_observable);
  void crossover(LinkedBinaryTree& other, mt19937& rng, int maxDepth);




 protected:                                        // local utilities
  void preorder(Node* v, PositionList& pl) const;  // preorder utility
  Node* copyPreOrder(const Node* root);
  double score;     // mean reward over 20 episodes
  double steps;     // mean steps-per-episode over 20 episodes
  long generation;  // which generation was tree "born"
 private:
  Node* _root;  // pointer to the root
};

// Global memory cell for GP (used by memory operations)
static double gpMemory = 0.0;

// Global flag to indicate partially observable mode.
// (Set this in main based on your experiment.)
static bool gPartiallyObservable = false;



// add the tree rooted at node child as this tree's left child
void LinkedBinaryTree::addLeftChild(const Position& p, const Node* child) {
  Node* v = p.v;
  v->left = copyPreOrder(child);  // deep copy child
  v->left->par = v;
}

// add the tree rooted at node child as this tree's right child
void LinkedBinaryTree::addRightChild(const Position& p, const Node* child) {
  Node* v = p.v;
  v->right = copyPreOrder(child);  // deep copy child
  v->right->par = v;
}

void LinkedBinaryTree::addLeftChild(const Position& p) {
  Node* v = p.v;
  v->left = new Node;
  v->left->par = v;
}

void LinkedBinaryTree::addRightChild(const Position& p) {
  Node* v = p.v;
  v->right = new Node;
  v->right->par = v;
}

// return a list of all nodes
LinkedBinaryTree::PositionList LinkedBinaryTree::positions() const {
  PositionList pl;
  preorder(_root, pl);
  return PositionList(pl);
}

void LinkedBinaryTree::preorder(Node* v, PositionList& pl) const {
  pl.push_back(Position(v));
  if (v->left != NULL) preorder(v->left, pl);
  if (v->right != NULL) preorder(v->right, pl);
}

int LinkedBinaryTree::size(Node* v) const {
  int lsize = 0;
  int rsize = 0;
  if (v->left != NULL) lsize = size(v->left);
  if (v->right != NULL) rsize = size(v->right);
  return 1 + lsize + rsize;
}

int LinkedBinaryTree::depth() const {
  PositionList pl = positions();
  int depth = 0;
  for (auto& p : pl) depth = std::max(depth, p.v->depth());
  return depth;
}

LinkedBinaryTree::Node* LinkedBinaryTree::copyPreOrder(const Node* root) {
  if (root == NULL) return NULL;
  Node* nn = new Node;
  nn->elt = root->elt;
  nn->left = copyPreOrder(root->left);
  if (nn->left != NULL) nn->left->par = nn;
  nn->right = copyPreOrder(root->right);
  if (nn->right != NULL) nn->right->par = nn;
  return nn;
}

void LinkedBinaryTree::printExpression(Node* v) {
  if (v == nullptr)
    return;
  // Leaf node (terminal): simply print the element.
  if (v->left == nullptr && v->right == nullptr) {
    cout << v->elt;
    return;
  }
  // Special handling for unary operator "abs".
  if (v->elt == "abs") {
    cout << "abs(";
    printExpression(v->left);
    cout << ")";
    return;
  }
  // For binary operators, enclose the expression in parentheses.
  cout << "(";
  printExpression(v->left);
  cout << v->elt;
  printExpression(v->right);
  cout << ")";
}

double evalOp(string op, double x, double y = 0) {
  double result;
  if (op == "+")
    result = x + y;
  else if (op == "-")
    result = x - y;
  else if (op == "*")
    result = x * y;
  else if (op == "/") {
    result = x / y;
  } else if (op == ">") {
    result = x > y ? 1 : -1;
  } else if (op == "abs") {
    result = abs(x);
  } else
    result = 0;
  return isnan(result) || !isfinite(result) ? 0 : result;
}

double LinkedBinaryTree::evaluateExpression(const Position& p, double a, double b) {
  // Handle memory operations first
  if (p.v->elt == "READ_MEM")
    return gpMemory;
  if (p.v->elt == "WRITE_MEM") {
    double val = evaluateExpression(p.left(), a, b);
    gpMemory = val;
    return val;
  }

  // Standard evaluation
  if (!p.isExternal()) {
    auto x = evaluateExpression(p.left(), a, b);
    if (arity(p.v->elt) > 1) {
      auto y = evaluateExpression(p.right(), a, b);
      return evalOp(p.v->elt, x, y);
    } else {
      return evalOp(p.v->elt, x);
    }
  } else {
    if (p.v->elt == "a")
      return a;
    else if (p.v->elt == "b")
      return b;
    else
      return stod(p.v->elt);
  }
}

//=====================
/*
void LinkedBinaryTree::deleteSubtreeMutator(mt19937& rng) {
    // Get all positions in the tree.
    PositionList pl = positions();
    if (pl.size() <= 1) return;  // nothing to delete if only the root exists

    // Pick a random non-root node.
    int randIndex = randInt(rng, 1, pl.size() - 1);
    Node* target = pl[randIndex].v;

    // Define a simple recursive function to free a subtree.
    auto freeSubtree = [&](Node* node, auto&& freeSubtreeRef) -> void {
        if (node == nullptr) return;
        freeSubtreeRef(node->left, freeSubtreeRef);
        freeSubtreeRef(node->right, freeSubtreeRef);
        delete node;
    };

    // Delete both children of the target node.
    freeSubtree(target->left, freeSubtree);
    freeSubtree(target->right, freeSubtree);
    target->left = nullptr;
    target->right = nullptr;

    // Replace target's element with a random terminal to ensure it's valid.
    int choice = randInt(rng, 0, 2);
    if (choice == 0)
        target->elt = "a";
    else if (choice == 1)
        target->elt = "b";
    else {
        double val = (randDouble(rng) * 2) - 1;
        target->elt = to_string(val);
    }
}
*/

//===================================================

void LinkedBinaryTree::deleteSubtreeMutator(mt19937& rng) {
  PositionList pl = positions();
  // If only the root exists, there’s nothing to delete.
  if (pl.size() <= 1) return;

  vector<Node*> candidates;
  for (auto pos : pl) {
    if (!pos.isRoot())
      candidates.push_back(pos.v);
  }
  if (candidates.empty()) return;

  int idx = randInt(rng, 0, candidates.size() - 1);
  Node* target = candidates[idx];
  Node* parent = target->par;
  if (parent == nullptr) return;  // safety check

  // Create a new terminal node ("a") to replace the deleted subtree.
  Node* newNode = new Node;
  newNode->elt = "a";
  newNode->par = parent;

  if (parent->left == target)
    parent->left = newNode;
  else if (parent->right == target)
    parent->right = newNode;

  // Free the removed subtree.
  std::function<void(Node*)> freeSubtree = [&](Node* node) {
    if (node == nullptr)
      return;
    freeSubtree(node->left);
    freeSubtree(node->right);
    delete node;
  };
  freeSubtree(target);
}



//===============================================================================================================





void LinkedBinaryTree::addSubtreeMutator(mt19937& rng, const int maxDepth, bool partially_observable) {
  if (_root == NULL) return;

  // Get all positions in the tree.
  PositionList pl = positions();
  if (pl.empty()) return;

  // Pick a random node.
  int randIndex = randInt(rng, 0, pl.size() - 1);
  Node* nodeToMutate = pl[randIndex].v;

  // Save the original element.
  string originalElt = nodeToMutate->elt;

  // Choose a random operator based on whether we are in partially observable mode.
  vector<string> operators;
  if (partially_observable) {
    operators = {"+", "-", "*", "/", ">", "abs", "READ_MEM", "WRITE_MEM"};
  } else {
    operators = {"+", "-", "*", "/", ">", "abs"};
  }
  int opIndex = randInt(rng, 0, operators.size() - 1);
  string op = operators[opIndex];

  // Replace the node's element with the new operator.
  nodeToMutate->elt = op;

  // If the node was terminal, attach a left child with the original element.
  if (nodeToMutate->left == NULL) {
    addLeftChild(Position(nodeToMutate));
    nodeToMutate->left->elt = originalElt;
  }

  // If the operator is binary and there is no right child, add one.
  if (arity(op) > 1 && nodeToMutate->right == NULL) {
    addRightChild(Position(nodeToMutate));
    int choice = randInt(rng, 0, 2);
    if (choice == 0)
      nodeToMutate->right->elt = "a";
    else if (choice == 1)
      nodeToMutate->right->elt = "b";
    else {
      double val = (randDouble(rng) * 2) - 1;
      nodeToMutate->right->elt = to_string(val);
    }
  }
}


//===============================================================================================================


void LinkedBinaryTree::crossover(LinkedBinaryTree& other, mt19937& rng, int maxDepth) {
  // Gather all nodes in *this
  PositionList plA = positions();
  // Gather all nodes in 'other'
  PositionList plB = other.positions();

  // If either tree is too small to pick a subtree, skip
  if (plA.size() < 2 || plB.size() < 2) return;

  // Pick a random non-root node in each tree
  int idxA = randInt(rng, 1, plA.size() - 1);  // skip root index 0
  int idxB = randInt(rng, 1, plB.size() - 1);

  Node* nodeA = plA[idxA].v;
  Node* nodeB = plB[idxB].v;
  Node* parentA = nodeA->par;
  Node* parentB = nodeB->par;
  if (!parentA || !parentB) return; // safety check

  // Swap child pointers
  if (parentA->left == nodeA)
    parentA->left = nodeB;
  else
    parentA->right = nodeB;

  if (parentB->left == nodeB)
    parentB->left = nodeA;
  else
    parentB->right = nodeA;

  // Swap parent pointers
  Node* temp = nodeA->par;
  nodeA->par = nodeB->par;
  nodeB->par = temp;

  // Depth check: if either tree is now too deep, undo swap
  if (depth() > maxDepth || other.depth() > maxDepth) {
    // Re-swap them back
    // Re-attach nodeA to parentA
    if (parentA->left == nodeB)
      parentA->left = nodeA;
    else
      parentA->right = nodeA;

    if (parentB->left == nodeA)
      parentB->left = nodeB;
    else
      parentB->right = nodeB;

    // Swap parent pointers back
    temp = nodeA->par;
    nodeA->par = nodeB->par;
    nodeB->par = temp;
  }
}

LinkedBinaryTree pickParentTournament(const std::vector<LinkedBinaryTree>& pop,
                                      std::mt19937& rng,
                                      int survivors,
                                      int k = 3)
{
  // Pick the first candidate
  LinkedBinaryTree best = pop[randInt(rng, 0, survivors - 1)];
  // Compare with (k-1) more random candidates
  for (int i = 1; i < k; i++) {
    LinkedBinaryTree candidate = pop[randInt(rng, 0, survivors - 1)];
    if (candidate.getScore() > best.getScore()) {
      best = candidate;
    }
  }
  return best;
}




bool operator<(const LinkedBinaryTree& x, const LinkedBinaryTree& y) {
  return x.getScore() < y.getScore();
}

LinkedBinaryTree createExpressionTree(string postfix) {
  stack<LinkedBinaryTree> tree_stack;
  stringstream ss(postfix);
  // Split each line into words
  string token;
  while (getline(ss, token, ' ')) {
    LinkedBinaryTree t;
    if (!isOp(token)) {
      t.addRoot(token);
      tree_stack.push(t);
    } else {
      t.addRoot(token);
      if (arity(token) > 1) {
        LinkedBinaryTree r = tree_stack.top();
        tree_stack.pop();
        t.addRightChild(t.root(), r.root());
      }
      LinkedBinaryTree l = tree_stack.top();
      tree_stack.pop();
      t.addLeftChild(t.root(), l.root());
      tree_stack.push(t);
    }
  }
  return tree_stack.top();
}

void LinkedBinaryTree::randomExpressionTree(Node* p, const int& maxDepth, mt19937& rng) {
  if (p == NULL) return;

  if (maxDepth <= 0) {
    // Terminal: choose between "a", "b", or a constant.
    int choice = randInt(rng, 0, 2);
    if (choice == 0)
      p->elt = "a";
    else if (choice == 1)
      p->elt = "b";
    else {
      double val = (randDouble(rng) * 2) - 1;
      p->elt = to_string(val);
    }
    return;
  }

  // Decide whether to create an operator node.
  bool createOperator = (maxDepth > 1) ? randChoice(rng) : false;
  if (createOperator) {
    vector<string> ops;
    if (gPartiallyObservable)
      ops = {"+", "-", "*", "/", ">", "abs", "READ_MEM", "WRITE_MEM"};
    else
      ops = {"+", "-", "*", "/", ">", "abs"};

    int opIndex = randInt(rng, 0, ops.size() - 1);
    string op = ops[opIndex];
    p->elt = op;

    // Build left child.
    addLeftChild(Position(p));
    randomExpressionTree(p->left, maxDepth - 1, rng);

    // If binary operator, build right child.
    if (arity(op) > 1) {
      addRightChild(Position(p));
      randomExpressionTree(p->right, maxDepth - 1, rng);
    }
  } else {
    // Terminal node.
    int choice = randInt(rng, 0, 2);
    if (choice == 0)
      p->elt = "a";
    else if (choice == 1)
      p->elt = "b";
    else {
      double val = (randDouble(rng) * 2) - 1;
      p->elt = to_string(val);
    }
  }
}


//===================================================

LinkedBinaryTree createRandExpressionTree(int max_depth, mt19937& rng, bool partially_observable) {
  if (max_depth < 1) max_depth = 1;

  LinkedBinaryTree t;
  t.addRoot();

  // Set the global flag based on the parameter.
  gPartiallyObservable = partially_observable;

  // Build a random tree with the given depth.
  t.randomExpressionTree(t.root(), max_depth, rng);

  return t;
}


// evaluate tree t in the cart centering task
void evaluate(mt19937& rng, LinkedBinaryTree& t, const int& num_episode,
              bool animate, bool partially_observable = false) {
  cartCentering env;
  double mean_score = 0.0;
  double mean_steps = 0.0;
  for (int i = 0; i < num_episode; i++) {
    double episode_score = 0.0;
    int episode_steps = 0;
    env.reset(rng);
    while (!env.terminal()) {
      int action;
      if (partially_observable) {
        action = t.evaluateExpression(env.getCartXPos(), 0.0);
      } else {
        action = t.evaluateExpression(env.getCartXPos(), env.getCartXVel());
      }
      episode_score += env.update(action, animate);
      episode_steps++;
    }
    mean_score += episode_score;
    mean_steps += episode_steps;
  }
  t.setScore(mean_score / num_episode);
  t.setSteps(mean_steps / num_episode);
}

#include <cmath> // for std::fabs

struct LexLessThan {
  bool operator()(const LinkedBinaryTree& x, const LinkedBinaryTree& y) const {
    double scoreDiff = fabs(x.getScore() - y.getScore());
    if (scoreDiff < 0.01) {
      // Favor simpler trees (fewer nodes) when scores are similar
      return x.size() > y.size();
    } else {
      // Otherwise, compare by score
      return x.getScore() < y.getScore();
    }
  }
};



#include <fstream>  // For file output

int main() {
  mt19937 rng(42);
  // Experiment parameters
  const int NUM_TREE = 50;
  const int MAX_DEPTH_INITIAL = 1;
  const int MAX_DEPTH = 20;
  const int NUM_EPISODE = 20;
  const int MAX_GENERATIONS = 100;
  const bool PARTIALLY_OBSERVABLE = false; // For Part 2, this is false

  // --- Experiment 1: Default Comparator (using operator<) ---
  vector<LinkedBinaryTree> treesDefault;
  for (int i = 0; i < NUM_TREE; i++) {
    LinkedBinaryTree t = createRandExpressionTree(MAX_DEPTH_INITIAL, rng, PARTIALLY_OBSERVABLE);
    treesDefault.push_back(t);
  }
  vector<int> bestSizeDefault(MAX_GENERATIONS, 0);
  for (int g = 1; g <= MAX_GENERATIONS; g++) {
    // Evaluate each tree in the population
    for (auto &t : treesDefault) {
      if (t.getGeneration() < g - 1) continue;
      evaluate(rng, t, NUM_EPISODE, false, PARTIALLY_OBSERVABLE);
    }
    // Sort using the default overloaded operator<
    std::sort(treesDefault.begin(), treesDefault.end());
    // Remove worst 50% of trees
    treesDefault.erase(treesDefault.begin(), treesDefault.begin() + NUM_TREE / 2);
    // Best tree is the last element
    LinkedBinaryTree best = treesDefault.back();
    bestSizeDefault[g - 1] = best.size();
    // Replenish the population via mutation
    while (treesDefault.size() < NUM_TREE) {
      LinkedBinaryTree parent = treesDefault[randInt(rng, 0, (NUM_TREE / 2) - 1)];
      LinkedBinaryTree child(parent);
      child.setGeneration(g);
      child.deleteSubtreeMutator(rng);
      // Call addSubtreeMutator with third parameter false (memory off)
      child.addSubtreeMutator(rng, MAX_DEPTH, false);
      treesDefault.push_back(child);
    }
  }

  // --- Experiment 2: LexLessThan Comparator ---
  vector<LinkedBinaryTree> treesLex;
  for (int i = 0; i < NUM_TREE; i++) {
    LinkedBinaryTree t = createRandExpressionTree(MAX_DEPTH_INITIAL, rng, PARTIALLY_OBSERVABLE);
    treesLex.push_back(t);
  }
  vector<int> bestSizeLex(MAX_GENERATIONS, 0);
  for (int g = 1; g <= MAX_GENERATIONS; g++) {
    for (auto &t : treesLex) {
      if (t.getGeneration() < g - 1) continue;
      evaluate(rng, t, NUM_EPISODE, false, PARTIALLY_OBSERVABLE);
    }
    // Sort using LexLessThan comparator
    std::sort(treesLex.begin(), treesLex.end(), LexLessThan());
    treesLex.erase(treesLex.begin(), treesLex.begin() + NUM_TREE / 2);
    LinkedBinaryTree best = treesLex.back();
    bestSizeLex[g - 1] = best.size();
    while (treesLex.size() < NUM_TREE) {
      LinkedBinaryTree parent = treesLex[randInt(rng, 0, (NUM_TREE / 2) - 1)];
      LinkedBinaryTree child(parent);
      child.setGeneration(g);
      child.deleteSubtreeMutator(rng);
      // Use the three-parameter version, passing false
      child.addSubtreeMutator(rng, MAX_DEPTH, false);
      treesLex.push_back(child);
    }
  }

  // --- Write CSV file for comparison ---
  std::ofstream outFile("comparison.csv");
  outFile << "generation,default_size,lex_size\n";
  for (int g = 1; g <= MAX_GENERATIONS; g++) {
    outFile << g << "," << bestSizeDefault[g - 1] << "," << bestSizeLex[g - 1] << "\n";
  }
  outFile.close();

  std::cout << "Experiments complete. Results saved to comparison.csv\n";
  return 0;
}

