#include "Inductive.h"

#include "Bounds.h"
#include "ConciseCasts.h"
#include "Error.h"
#include "ExprUsesVar.h"
#include "Function.h"
#include "IR.h"
#include "IREquality.h"
#include "IRVisitor.h"
#include "ModulusRemainder.h"
#include "Simplify.h"
#include "Substitute.h"
#include <algorithm>
#include <map>

namespace Halide {
namespace Internal {

using std::map;
using std::pair;
using std::string;
using std::vector;

namespace {

class BaseCaseSolver : public IRVisitor {
    using IRVisitor::visit;
    const vector<string> &vars;
    const string &func;

    const vector<Interval> &start_box;
    const vector<bool> &is_inductive_var;
    const bool is_update;

    vector<Interval> condition_intervals;

    Scope<Interval> bounds;

    // Names of the inductive vars, for testing whether a select's condition
    // depends on any of them.
    Scope<> inductive_vars;

    int nested_select = 0;

    void visit(const Call *op) override {
        if (op->is_intrinsic(Call::if_then_else)) {

            // A select whose condition doesn't depend on any inductive var
            // (e.g. a plain dispatch on a small non-inductive index, like a
            // "case" switch) can't possibly be the select that divides the
            // inductive recursion into a base case and a recursive case, so
            // it doesn't count towards the "nested select" restriction, and
            // doesn't need to restrict the inductive vars' intervals.
            bool cond_is_inductive = expr_uses_vars(op->args[0], inductive_vars);
            if(!cond_is_inductive) {
                IRVisitor::visit(op);
                return;
            }

            // Theoretically there is no need to check op->args[0].
            // Select nodes are only converted to if_then_else when the condition is pure,
            // which means the condition cannot have any recursive calls.
            // std::cout<<"cond is" << op->args[0];
            op->args[0].accept(this);

            bool left_recurse = false, right_recurse = false;
            visit_with(op->args[1], [&](auto *self, const Call *inner_op) {
                if (inner_op->name == func) {
                    left_recurse = true;
                }
                self->visit_base(inner_op);
            });
            visit_with(op->args[2], [&](auto *self, const Call *inner_op) {
                if (inner_op->name == func) {
                    right_recurse = true;
                }
                self->visit_base(inner_op);
            });
            // Again, this check is theoretically unnecessary
            user_assert(!(left_recurse && right_recurse)) << "Select node " << op->args[1] << op->args[2] << " in inductive function " << func << " does not have a base case";

            

            if (cond_is_inductive) {
                nested_select += 1;
            }
            vector<Interval> old_intervals = condition_intervals;
            if (left_recurse) {
                if (cond_is_inductive) {
                    for (size_t i = 0; i < vars.size(); i++) {
                        Interval inter = Interval::make_intersection(old_intervals[i], solve_for_outer_interval(simplify(op->args[0]), vars[i]));
                        condition_intervals[i] = Interval(inter.min, Interval::pos_inf());
                        bounds.push(vars[i], condition_intervals[i]);
                    }
                }
                op->args[1].accept(this);
                if (cond_is_inductive) {
                    for (const auto &var : vars) {
                        bounds.pop(var);
                    }
                }
            }
            if (right_recurse) {
                if (cond_is_inductive) {
                    for (size_t i = 0; i < vars.size(); i++) {
                        Interval inter = Interval::make_intersection(old_intervals[i], solve_for_outer_interval(simplify(!op->args[0]), vars[i]));
                        condition_intervals[i] = Interval(inter.min, Interval::pos_inf());
                        bounds.push(vars[i], condition_intervals[i]);
                    }
                }
                op->args[2].accept(this);
                if (cond_is_inductive) {
                    for (const auto &var : vars) {
                        bounds.pop(var);
                    }
                }
            }
            condition_intervals = old_intervals;
            if (cond_is_inductive) {
                nested_select -= 1;
            }
        } else if (op->name == func) {
            fprintf(stderr, "DEBUG: BaseCaseSolver visiting self-call to %s, nested_select=%d\n", func.c_str(), nested_select);

            // An update definition's self-reference whose args match the
            // pure vars exactly, position for position, is an ordinary
            // reduction accumulator (e.g. the RDom running-max idiom used
            // alongside a genuinely shifted recursive step in the same
            // update) -- it isn't a recursive step at all, so it's exempt
            // from both the "no nested select" restriction and the
            // monotonic-decrease proof below. This exemption only applies
            // within update definitions: a pure definition's own
            // self-reference (the classic inductive-pure-def case) must
            // still be a genuine shift.
            if (is_update) {
                bool exact_pure_self_reference = true;
                for (size_t position = 0; position < vars.size() && exact_pure_self_reference; position++) {
                    if (position >= op->args.size() ||
                        !equal(op->args[position], Variable::make(op->args[position].type(), vars[position]))) {
                        exact_pure_self_reference = false;
                    }
                }
                if (exact_pure_self_reference) {
                    fprintf(stderr, "DEBUG: self-call to %s is an exact pure self-reference (accumulator), exempt\n", func.c_str());
                    IRVisitor::visit(op);
                    return;
                }
            }

            user_assert(nested_select > 0) << "Function " << func << " contains an inductive function reference outside of a select operation value.\n";
            user_assert(nested_select == 1) << "Function " << func << " contains an inductive function reference inside a nested select operation.\n";
            bool found_inductive = false;
            for (size_t position = 0; position < vars.size(); position++) {
                const Expr inductive_expr = op->args[position];
                if (!is_inductive_var[position]) {
                    // Not an inductive dimension (e.g. a literal case/stage
                    // index in a switch-style dispatch) -- there's no
                    // monotonic-decrease requirement to prove here; just
                    // make sure the required region includes whatever
                    // concrete value(s) this call actually references.
                    Interval new_interval;
                    if (equal(Variable::make(inductive_expr.type(), vars[position]), inductive_expr)) {
                        // Pass-through dimension: the call doesn't shift
                        // this position at all (e.g. `y` in
                        // g(max(0, x-1), y)). condition_intervals/bounds
                        // carry no information about a dimension the
                        // select's condition never mentions, so intersecting
                        // against them (as below) can spuriously produce an
                        // unbounded interval. The already-known required
                        // region for this dimension (start_box) is exactly
                        // right here, just as the inductive-var branch below
                        // uses it for its own unshifted case.
                        new_interval = start_box[position];
                    } else {
                        new_interval = Interval::make_intersection(
                            bounds_of_expr_in_scope(inductive_expr, bounds), condition_intervals[position]);
                    }
                    result_intervals[position] = Interval::make_union(result_intervals[position], new_interval);
                    // A recursive call that changes a non-inductive
                    // dimension (e.g. steps from stage r=1 to stage r=0)
                    // is itself evidence that this call is a legitimate
                    // step of a well-founded recursion: that dimension's
                    // own range is already finite and bounded (handled
                    // above, not via monotonicity), so no interval proof
                    // is needed here to know it can't recurse forever.
                    if (!equal(Variable::make(inductive_expr.type(), vars[position]), inductive_expr)) {
                        found_inductive = true;
                    }
                    continue;
                }
                const Expr new_v = Variable::make(inductive_expr.type(), vars[position]);
                const Expr gets_lower = simplify(new_v - inductive_expr > 0, bounds);
                const Interval i_lower = solve_for_inner_interval(gets_lower, vars[position]);
                {
                    auto to_s = [](const Expr &e) { std::ostringstream oss; oss << e; return oss.str(); };
                    fprintf(stderr, "DEBUG: pos=%zu var=%s inductive_expr=%s gets_lower=%s i_lower=[%s,%s] cond_interval=[%s,%s] start_box=[%s,%s]\n",
                            position, vars[position].c_str(), to_s(inductive_expr).c_str(),
                            to_s(gets_lower).c_str(),
                            to_s(i_lower.min).c_str(), to_s(i_lower.max).c_str(),
                            to_s(condition_intervals[position].min).c_str(), to_s(condition_intervals[position].max).c_str(),
                            to_s(start_box[position].min).c_str(), to_s(start_box[position].max).c_str());
                }

                Interval new_interval;
                if (equal(new_v, inductive_expr)) {
                    new_interval = start_box[position];
                } else if (i_lower.is_everything()) {
                    found_inductive = true;
                    new_interval = Interval(Interval::neg_inf(), start_box[position].max);
                } else {
                    std::ostringstream err;
                    err << "Inductive variable " << vars[position] << " in inductive function " << func << " is not provably monotonically decreasing outside of the base case.";
                    user_error <<(Expr)op<< err.str() << "\n";
                }
                new_interval = Interval::make_intersection(new_interval, condition_intervals[position]);
                Scope<Interval> i_scope;
                i_scope.push(vars[position], new_interval);
                result_intervals[position] = Interval::make_union(result_intervals[position], Interval::make_union(new_interval, bounds_of_expr_in_scope(inductive_expr, i_scope)));
            }
            user_assert(found_inductive) << Expr(op)<<"Unable to prove in inductive function " << func << " that the inductive step is monotonically decreasing.\n";

            IRVisitor::visit(op);

        } else {
            IRVisitor::visit(op);
        }
    }

public:
    vector<Interval> result_intervals;

    BaseCaseSolver(const vector<string> &v, const vector<bool> &is_inductive_var, const string &func, const vector<Interval> &con, bool is_update)
        : vars(v), func(func), start_box(con), is_inductive_var(is_inductive_var), is_update(is_update) {
        condition_intervals = vector<Interval>(start_box.size());
        result_intervals = vector<Interval>(start_box.size(), Interval::nothing());
        for (size_t i = 0; i < vars.size(); i++) {
            debug(0) << vars[i] << is_inductive_var[i];
            if (is_inductive_var[i]) {
                inductive_vars.push(vars[i]);
            }
        }
    }
};

// Finds every constant shift applied to var_name across recursive
// self-references to func_name, for split_gcd.
class FindDiffs : public IRVisitor {
    using IRVisitor::visit;
    const string &var_name;
    const int pos;
    const string &func_name;
    vector<int64_t> &diffs;

    void visit(const Call *op) override {
        if (op->name == func_name) {
            const Expr &arg = op->args[pos];
            int64_t diff = 1;
            if (const Sub *sub = arg.as<Sub>()) {
                const Variable *v_lhs = sub->a.as<Variable>();
                if (v_lhs && v_lhs->name == var_name) {
                    if (const IntImm *imm = sub->b.as<IntImm>()) {
                        diff = imm->value;
                    }
                }
            } else if (const Add *add = arg.as<Add>()) {
                const Variable *v_lhs = add->a.as<Variable>();
                const Variable *v_rhs = add->b.as<Variable>();
                if (v_lhs && v_lhs->name == var_name) {
                    if (const IntImm *imm = add->b.as<IntImm>()) {
                        diff = imm->value;
                    }
                } else if (v_rhs && v_rhs->name == var_name) {
                    if (const IntImm *imm = add->a.as<IntImm>()) {
                        diff = imm->value;
                    }
                }
            } else if (const Variable *v = arg.as<Variable>()) {
                if (v->name == var_name) {
                    diff = 0;
                }
            }
            if (diff != 0) {
                diffs.push_back(diff);
            }
        }
        IRVisitor::visit(op);
    }

public:
    FindDiffs(const string &v, int p, const string &f, vector<int64_t> &d)
        : var_name(v), pos(p), func_name(f), diffs(d) {
    }
};

// Checks, for the recursive self-reference at position `selpos` in a call,
// whether some earlier-appearing dimension of that same call is provably
// monotonically decreasing, making the relative evaluation order of the
// dimension at selpos irrelevant to correctness.
class InductiveOrderChecker : public IRVisitor {
    using IRVisitor::visit;
    const vector<string> &vars;
    const string &func;
    const int &selpos;

    void visit(const Call *op) override {
        if (op->name == func) {
            bool is_pure_local = false;
            for (size_t position = 0; position < vars.size(); position++) {
                if ((int)position == selpos) {
                    continue;
                }
                const Expr new_v = Variable::make(op->args[position].type(), vars[position]);
                const Expr gets_lower = simplify(new_v - op->args[position] > 0);
                if (can_prove(gets_lower)) {
                    is_pure_local = true;
                    break;
                }
            }
            is_pure &= is_pure_local;
        }
        IRVisitor::visit(op);
    }

public:
    bool is_pure = true;
    InductiveOrderChecker(const vector<string> &v, const string &func, const int &selpos)
        : vars(v), func(func), selpos(selpos) {
    }
};

}  // anonymous namespace

int split_gcd(const Function &fn, const string &var_name) {
    int pos = -1;
    for (size_t i = 0; i < fn.args().size(); i++) {
        if (fn.args()[i] == var_name) {
            pos = (int)i;
            break;
        }
    }
    user_assert(pos != -1) << "Variable " << var_name << " not found in function " << fn.name() << "\n";

    vector<int64_t> diffs;
    FindDiffs find_diffs(var_name, pos, fn.name(), diffs);
    for (const Expr &e : fn.definition().values()) {
        e.accept(&find_diffs);
    }
    if (diffs.empty()) {
        return 1;
    }

    int64_t result = 0;
    for (int64_t d : diffs) {
        result = gcd(result, std::abs(d));
    }
    return (int)result;
}

bool can_be_pure(const vector<string> &vars, const Function &fn, const int &selpos) {
    InductiveOrderChecker checker(vars, fn.name(), selpos);
    for (const Expr &e : fn.definition().values()) {
        e.accept(&checker);
    }
    return checker.is_pure;
}

bool splits_reordered(const vector<string> &vars, const Function &fn) {
    // Map each original variable to an ordered chain of its split
    // variables, from innermost to outermost.
    map<string, vector<string>> split_var_map;
    for (const auto &split : fn.definition().schedule().splits()) {
        bool found = false;
        for (auto &it : split_var_map) {
            auto pos = std::find(it.second.begin(), it.second.end(), split.old_var);
            if (pos != it.second.end()) {
                it.second.erase(pos);
                it.second.insert(pos, split.outer);
                it.second.insert(pos, split.inner);
                found = true;
                break;
            }
        }
        if (!found) {
            split_var_map[split.old_var] = {split.inner, split.outer};
        }
    }

    const vector<Dim> &dims = fn.definition().schedule().dims();
    bool reordered = false;
    for (const auto &var : vars) {
        auto it = split_var_map.find(var);
        if (it == split_var_map.end()) {
            continue;
        }
        const vector<string> &split_vars = it->second;
        for (size_t i = 1; i < split_vars.size(); i++) {
            int inner_pos = -1, outer_pos = -1;
            for (size_t d = 0; d < dims.size(); d++) {
                if (dims[d].var == split_vars[i - 1]) {
                    inner_pos = (int)d;
                }
                if (dims[d].var == split_vars[i]) {
                    outer_pos = (int)d;
                }
            }
            // Dims are ordered innermost-first, so the pure (inner) part
            // of a split inductive dimension must have a strictly smaller
            // index than its inductive (outer) part.
            if (inner_pos != -1 && outer_pos != -1 && inner_pos > outer_pos) {
                int selpos = -1;
                for (size_t i2 = 0; i2 < fn.args().size(); i2++) {
                    if (fn.args()[i2] == var) {
                        selpos = (int)i2;
                        break;
                    }
                }
                if (selpos == -1 || !can_be_pure(fn.args(), fn, selpos)) {
                    reordered = true;
                }
            }
        }
    }
    return reordered;
}

Box expand_to_include_base_case(const vector<string> &vars, const vector<bool> &is_inductive_var,
                                const Expr &RHS, const string &func, const Box &box_required,
                                bool is_update) {
    {
        std::ostringstream oss;
        oss << RHS;
        fprintf(stderr, "DEBUG: expand_to_include_base_case called for func=%s RHS=%s\n", func.c_str(), oss.str().c_str());
    }
    Expr substed = substitute_in_all_lets(RHS);
    Box box2 = box_required;
    BaseCaseSolver b(vars, is_inductive_var, func, box_required.bounds, is_update);
    substed.accept(&b);
    for (size_t i = 0; i < vars.size(); i++) {
        user_assert(b.result_intervals[i].is_bounded() || b.result_intervals[i].is_empty()) << "Unable to prove that the inductive function " << func << " uses a bounded interval";
        if (!b.result_intervals[i].is_empty()) {
            Interval new_interval(min(b.result_intervals[i].min, box_required[i].min), box_required[i].max);
            box2[i] = new_interval;
        }
    }

    return box2;
}

}  // namespace Internal
}  // namespace Halide
