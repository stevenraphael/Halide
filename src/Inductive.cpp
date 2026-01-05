#include "Inductive.h"

#include "Bounds.h"
#include "ConciseCasts.h"
#include "Error.h"
#include "Function.h"
#include "IR.h"
#include "IREquality.h"
#include "IRVisitor.h"
#include "Simplify.h"
#include "Substitute.h"

namespace Halide {
namespace Internal {

using std::string;
using std::vector;
using std::pair;
using std::map;

class BaseCaseSolver : public IRVisitor {
    using IRVisitor::visit;
    const vector<string> &vars;
    const string &func;

    const vector<Interval> &start_box;

    vector<Interval> condition_intervals;

    Scope<Interval> bounds;

    int nested_select = 0;

    void visit(const Call *op) override {
        if (op->is_intrinsic(Call::if_then_else)) {
            nested_select += 1;
            vector<Interval> old_intervals = condition_intervals;
            for (size_t i = 0; i < vars.size(); i++) {
                condition_intervals[i] = Interval::make_intersection(old_intervals[i], solve_for_outer_interval(simplify(op->args[0]), vars[i]));
                bounds.push(vars[i], condition_intervals[i]);
            }

            op->args[1].accept(this);
            for (size_t i = 0; i < vars.size(); i++) {
                condition_intervals[i] = Interval::make_intersection(old_intervals[i], solve_for_outer_interval(simplify(!op->args[0]), vars[i]));
                bounds.pop(vars[i]);
                bounds.push(vars[i], condition_intervals[i]);
            }
            op->args[2].accept(this);
            condition_intervals = old_intervals;
            for (const auto &var : vars) {
                bounds.pop(var);
            }
            nested_select -= 1;
        } else if (op->name == func) {
            user_assert(nested_select > 0) << "Function " << func << " contains an inductive function reference outside of a select operation.\n";
            user_assert(nested_select == 1) << "Function " << func << " contains an inductive function reference inside a nested select operation.\n";
            bool found_inductive = false;
            for (size_t position = 0; position < vars.size(); position++) {
                const Expr inductive_expr = op->args[position];
                const Expr new_v = Variable::make(inductive_expr.type(), vars[position]);
                const Expr gets_lower = simplify(new_v - inductive_expr > 0, true, bounds);
                const Interval i_lower = solve_for_inner_interval(gets_lower, vars[position]);

                Interval new_interval;
                if (equal(new_v, inductive_expr)) {
                    new_interval = start_box[position];
                } else if (i_lower.is_everything()) {
                    found_inductive = true;
                    new_interval = Interval(Interval::neg_inf(), start_box[position].max);
                } else {
                    new_interval = Interval::everything();
                }
                new_interval = Interval::make_intersection(new_interval, condition_intervals[position]);
                Scope<Interval> i_scope;
                i_scope.push(vars[position], new_interval);
                result_intervals[position] = Interval::make_union(result_intervals[position], Interval::make_union(new_interval, bounds_of_expr_in_scope(inductive_expr, i_scope)));
            }
            user_assert(found_inductive) << "Unable to prove in inductive function " << func << " that the inductive step is monotonically decreasing.\n";

            IRVisitor::visit(op);

        } else {
            IRVisitor::visit(op);
        }
    }

public:
    vector<Interval> result_intervals;

    BaseCaseSolver(const vector<string> &v, const string &func, const vector<Interval> &con)
        : vars(v), func(func), start_box(con) {
        condition_intervals = vector<Interval>(start_box.size());
        result_intervals = vector<Interval>(start_box.size(), Interval::nothing());
    }
};

class InductiveOrderChecker : public IRVisitor {
    using IRVisitor::visit;
    const vector<string> &vars;
    const string &func;
    const int &selpos;
    const map<string, pair<int, int>> &var_appearances;

    void visit(const Call *op) override {
        if (op->name == func) {
            std::vector<string> pos_args;
            std::vector<string> neg_args;
            // attempt to prove that all dimensions earlier in the list are monotonically decreasing
            for (size_t position = 0; position < vars.size(); position++) {
                if(position != selpos){
                    auto it1 = var_appearances.find(vars[selpos]);
                    auto it2 = var_appearances.find(vars[position]);
                    if(it2->second.second >= it1->second.first){
                        bool is_pure_local = false;
                        const Expr inductive_expr = simplify(op->args[position]);
                        if (const Sub *sub = inductive_expr.as<Sub>()) {
                            const Variable *var = sub->a.as<Variable>();
                            const std::optional<int64_t> constant = as_const_int(sub->b);
                            if (var && constant && var->name == vars[position] && *constant > 0) {
                                is_pure_local = true;
                            }
                        }
                        is_pure |= is_pure_local;
                    }
                }
            }
        }
        IRVisitor::visit(op);
    }

public:
    bool is_pure = false;
    InductiveOrderChecker(const vector<string> &v, const string &func, const int &selpos, const map<string, pair<int, int>> &var_appearances)
        : vars(v), func(func), selpos(selpos), var_appearances(var_appearances) {
    }

};
    

// anonymous namespace





int split_gcd(const Function &fn, const string &var_name) {
    int pos;
    for (pos = 0; pos < (int)fn.args().size(); pos++) {
        if (fn.args()[pos] == var_name) {
            break;
        }
    }
    user_assert(pos < (int)fn.args().size()) << "Variable " << var_name << " not found in function " << fn.name() << "\n";
    const Expr &e = fn.values()[pos];
    vector<int64_t> diffs;

    struct FindDiffs : public IRVisitor {
        using IRVisitor::visit;
        const string &var_name;
        const int pos;
        const string &func_name;
        vector<int64_t> &diffs;
        FindDiffs(const string &v, int p, const string &f, vector<int64_t> &d) : var_name(v), pos(p), func_name(f), diffs(d) {
        }

        void visit(const Call *op) override {
            if(op->name == func_name) {
                const Expr &arg = op->args[pos];
                int diff = 1;
                if(const Sub *sub = arg.as<Sub>()) {
                    const Variable *v_lhs = sub->a.as<Variable>();
                    const Variable *v_rhs = sub->b.as<Variable>();
                    if (v_lhs && v_lhs->name == var_name) {
                        if (const IntImm *imm = sub->b.as<IntImm>()) {
                            diff = imm->value;
                        }
                    }
                }
                else if(const Add *add = arg.as<Add>()) {
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
                }
                else if(const Variable *v = arg.as<Variable>()) {
                    if (v->name == var_name) {
                        diff = 0;
                    }
                }
                if(diff != 0) {
                    diffs.push_back(diff);
                }

            }
        }
    };

    FindDiffs find_diffs(var_name, pos, fn.name(), diffs);
    for(const Expr &e : fn.values()){
        e.accept(&find_diffs);
    }
    if(diffs.size() == 0) {
        return 1;
    }

    int64_t result = 0;
    for (int64_t d : diffs) {
        result = gcd(result, std::abs(d));
    }
    return result;
}

bool can_be_pure(const std::vector<std::string> &vars, const Function &fn, const int &selpos) {
    // construct map of split variables to their original variable
    const vector<Split> splits = fn.definition().schedule().splits();
    std::map<std::string, std::string> split_var_map;
    for (const auto &split : fn.definition().schedule().splits()) {
        if (split_var_map.find(split.old_var) != split_var_map.end()) {
            split_var_map[split.inner] = split_var_map[split.old_var];
            split_var_map[split.outer] = split_var_map[split.old_var];
        } else {
            split_var_map[split.inner] = split.old_var;
            split_var_map[split.outer] = split.old_var;
        }
    }
    vector<Dim> dims = fn.definition().schedule().dims();
    map<string, pair<int, int>> var_appearances;
    for(const auto &var : vars) {
        int first_appearance = -1;
        int last_appearance = -1;
        for (size_t i = 0; i < dims.size(); i++) {
            if(!dims[i].is_inductive()) {
                continue;
            }
            const auto &dim = dims[i];
            auto it = split_var_map.find(dim.var);
            if (it != split_var_map.end() && it->second == var) {
                if (first_appearance == -1) {
                    first_appearance = i;
                }
                last_appearance = i;
            }
        }
        var_appearances[var] = {first_appearance, last_appearance};
    }

    InductiveOrderChecker checker(vars, fn.name(), selpos, var_appearances);
    for (size_t pos = 0; pos < fn.values().size(); pos++) {
        fn.values()[pos].accept(&checker);
    }
    return checker.is_pure;
}

}  // namespace Internal
}  // namespace Halide
