#!/usr/bin/env python3

"""
Simple test script to verify Memgraph installation and connectivity
"""

from neo4j import GraphDatabase
import sys

def test_memgraph_connection():
    """Test connection to Memgraph"""
    try:
        # Connection parameters (Memgraph uses same protocol as Neo4j)
        URI = "bolt://localhost:7687"
        AUTH = ("", "")  # Default - no authentication
        
        print("Connecting to Memgraph...")
        
        # Create driver instance
        driver = GraphDatabase.driver(URI, auth=AUTH)
        
        # Test connection
        with driver.session() as session:
            result = session.run("RETURN 'Hello Memgraph!' as message")
            message = result.single()["message"]
            print(f"✅ Connected! Message: {message}")
            
            # Test basic graph operations
            print("\nTesting basic graph operations...")
            
            # Create nodes
            session.run("CREATE (a:Person {name: 'Alice'})")
            session.run("CREATE (b:Person {name: 'Bob'})")
            
            # Create relationship
            session.run("""
                MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
                CREATE (a)-[:KNOWS]->(b)
            """)
            
            # Query the graph
            result = session.run("""
                MATCH (a:Person)-[:KNOWS]->(b:Person)
                RETURN a.name as person1, b.name as person2
            """)
            
            records = list(result)
            if records:
                record = records[0]
                print(f"✅ Graph query successful: {record['person1']} knows {record['person2']}")
            else:
                print("⚠️  Graph query returned no results")
            
            # Clean up
            session.run("MATCH (n) DETACH DELETE n")
            print("✅ Test data cleaned up")
        
        driver.close()
        print("\n🎉 All tests passed! Memgraph is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\n💡 Make sure Memgraph is running:")
        print("   Docker: docker run -p 7687:7687 -p 7444:7444 --name memgraph memgraph/memgraph-mage")
        print("   Or native: memgraph --log-level=TRACE")
        return False

if __name__ == "__main__":
    success = test_memgraph_connection()
    sys.exit(0 if success else 1)
