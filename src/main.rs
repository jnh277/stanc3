#![feature(box_syntax, box_patterns)]
#[macro_use] extern crate lalrpop_util;
pub mod ast;
pub mod interpret;

use ast::Expr;
//use ast::Expr::{Int};
use ast::BinOp;

lalrpop_mod!(pub grammar); // synthesized by LALRPOP

fn main() {
    println!("Hello, world!");
}

#[test]
fn num_terms() {
    assert!(grammar::TermParser::new().parse("22").is_ok());
    assert!(grammar::TermParser::new().parse("(22)").is_ok());
    assert!(grammar::TermParser::new().parse("((((22))))").is_ok());
    assert!(grammar::TermParser::new().parse("((22)").is_err());
}

#[test]
fn exprs_test() {
    let ep = grammar::ExprParser::new();
    assert!(ep.parse("22 + 23").is_ok());
    assert!(ep.parse("(22) * 23 + 21").is_ok());
    assert!(ep.parse("((((22.1)))) / 2").is_ok());
    match ep.parse("1 * 2 + 3") {
        Ok(box Expr::BinOp(box Expr::BinOp(i, BinOp::Mul, j), BinOp::Add, k))
            => println!("{:?} {:?} {:?}", i, j, k),
        Ok(other) => panic!("Precedence not respected; {:?}", other),
        Err(x) => panic!("Failed to parse! {:?}", x),
    }
    assert_eq!(interpret::evalScal(&ep.parse("2 + 3 * 4").unwrap()), 14.0);
    assert_eq!(interpret::evalScal(&ep.parse("2 * 4 - 1").unwrap()), 7.0);
    assert_eq!(interpret::evalScal(&ep.parse("2 * (4 - 1)").unwrap()), 6.0);
}
