# ("power_rule_differentiation", "calculus"),
# ("basic_algebra", "algebra"),
# ("log", "algebra"),
# ("fraction_to_decimal", "basic_math"),
# ("multiply_int_to_22_matrix", "algebra"),
# ("area_of_triangle", "geometry"),
# ("valid_triangle", "geometry"),
# ("prime_factors", "misc"),
# ("pythagorean_theorem", "geometry"),
# ("linear_equations", "algebra"),
# ("prime_factors", "misc"),
# ("fraction_multiplication", "basic_math"),
# ("angle_regular_polygon", "geometry"),
# ("combinations", "statistics"),
# ("factorial", "basic_math"),
# ("surface_area_cube", "geometry"),
# ("surface_area_cuboid", "geometry"),
# ("surface_area_cylinder", "geometry"),
# ("volume_cube", "geometry"),
# ("volume_cuboid", "geometry"),
# ("volume_cylinder", "geometry"),
# ("surface_area_cone", "geometry"),
# ("volume_cone", "geometry"),
# ("common_factors", "misc"),
# ("intersection_of_two_lines", "algebra"),
# ("permutation", "statistics"),
# ("vector_cross", "algebra"),
# ("compare_fractions", "basic_math"),
# ("simple_interest", "algebra"),
# ("matrix_multiplication", "algebra"),
# ("cube_root", "basic_math"),
# ("power_rule_integration", "calculus"),
# ("fourth_angle_of_quadrilateral", "geometry"),
# ("quadratic_equation", "algebra"),
# ("dice_sum_probability", "statistics"),
# ("exponentiation", "basic_math"),
# ("confidence_interval", "statistics"),
TOPICS = [
    # dict(subtopic="addition", topic="basic_math"),
    # dict(subtopic="subtraction", topic="basic_math"),
    # dict(subtopic="multiplication", topic="basic_math"),
    dict(subtopic="power_rule_differentiation", topic="calculus"),
    dict(subtopic="basic_algebra", topic="algebra"),
    dict(subtopic="log", topic="algebra"),
    dict(subtopic="fraction_to_decimal", topic="basic_math"),
    dict(subtopic="multiply_int_to_22_matrix", topic="algebra"),
    dict(subtopic="area_of_triangle", topic="geometry"),
    dict(subtopic="valid_triangle", topic="geometry"),
    dict(subtopic="prime_factors", topic="misc"),
    dict(subtopic="pythagorean_theorem", topic="geometry"),
    dict(subtopic="linear_equations", topic="algebra"),
    dict(subtopic="prime_factors", topic="misc"),
    dict(subtopic="fraction_multiplication", topic="basic_math"),
    dict(subtopic="angle_regular_polygon", topic="geometry"),
    dict(subtopic="combinations", topic="statistics"),
    dict(subtopic="factorial", topic="basic_math"),
    dict(subtopic="surface_area_cube", topic="geometry"),
    dict(subtopic="surface_area_cuboid", topic="geometry"),
    dict(subtopic="surface_area_cylinder", topic="geometry"),
    dict(subtopic="volume_cube", topic="geometry"),
    dict(subtopic="volume_cuboid", topic="geometry"),
    dict(subtopic="volume_cylinder", topic="geometry"),
    dict(subtopic="surface_area_cone", topic="geometry"),
    dict(subtopic="volume_cone", topic="geometry"),
    dict(subtopic="common_factors", topic="misc"),
    dict(subtopic="intersection_of_two_lines", topic="algebra"),
    dict(subtopic="permutation", topic="statistics"),
    dict(subtopic="vector_cross", topic="algebra"),
    dict(subtopic="compare_fractions", topic="basic_math"),
    dict(subtopic="simple_interest", topic="algebra"),
    dict(subtopic="matrix_multiplication", topic="algebra"),
    dict(subtopic="cube_root", topic="basic_math"),
    dict(subtopic="power_rule_integration", topic="calculus"),
    dict(subtopic="fourth_angle_of_quadrilateral", topic="geometry"),
    dict(subtopic="quadratic_equation", topic="algebra"),
    dict(subtopic="dice_sum_probability", topic="statistics"),
    dict(subtopic="exponentiation", topic="basic_math"),
    dict(subtopic="confidence_interval", topic="statistics"),
]


if __name__ == "__main__":
    import mathgenerator
    from latex2sympy2 import latex2sympy

    print(mathgenerator)
    for topic in TOPICS:
        subtopic = topic["subtopic"]
        topic = topic["topic"]
        atom_problem, atom_answer = eval(f"mathgenerator.{topic}.{subtopic}()")
        print(f"Topic: {topic}, Subtopic: {subtopic}")
        print(f"Generated atom math problem: {atom_problem}")
        print(f"Answer: {atom_answer}")
        atom_problem = atom_problem.replace("$", "").replace("=", "").strip()
        try:
            atom_answer = latex2sympy(atom_answer)
            print(f"Sympy Answer: {atom_answer}")
        except Exception:
            try:
                atom_answer = atom_answer.replace("$", "").replace("=", "").strip()
                atom_answer = eval(atom_answer)
                print(f"Python Answer: {atom_answer}")
            except Exception as e:
                print(f"Error: {e}")
        print("--" * 10)
