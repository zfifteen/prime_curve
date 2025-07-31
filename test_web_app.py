#!/usr/bin/env python3
"""
Simple test script to verify the Flask web application is working correctly.
"""

import requests
import json
import sys

def test_main_page():
    """Test that the main page loads without errors."""
    try:
        response = requests.get('http://127.0.0.1:5000/')
        if response.status_code == 200 and 'Prime Curve Interactive 3D Visualizations' in response.text:
            print("✓ Main page loads correctly")
            return True
        else:
            print(f"✗ Main page failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Main page failed: {e}")
        return False

def test_plotly_js():
    """Test that the plotly.js library is served correctly."""
    try:
        response = requests.get('http://127.0.0.1:5000/plotly.js')
        if response.status_code == 200 and 'plotly.js' in response.text:
            print("✓ Plotly.js library serves correctly")
            return True
        else:
            print(f"✗ Plotly.js failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Plotly.js failed: {e}")
        return False

def test_plot_endpoints():
    """Test that plot endpoints return valid JSON data."""
    views_to_test = [
        "Prime Gaps",
        "Zeta Function", 
        "Curvature Map",
        "Fourier Amplitudes",
        "GMM Density"
    ]
    
    all_passed = True
    for view in views_to_test:
        try:
            response = requests.get(f'http://127.0.0.1:5000/plot/{view}')
            if response.status_code == 200:
                data = response.json()
                # Check for required Plotly structure
                if 'data' in data and 'layout' in data and len(data['data']) > 0:
                    expected_title = f"3D Plot: {view}"
                    if data['layout']['title']['text'] == expected_title:
                        print(f"✓ {view} endpoint works correctly")
                    else:
                        print(f"✗ {view} title mismatch: {data['layout']['title']['text']}")
                        all_passed = False
                else:
                    print(f"✗ {view} missing required data structure")
                    all_passed = False
            else:
                print(f"✗ {view} endpoint failed: {response.status_code}")
                all_passed = False
        except Exception as e:
            print(f"✗ {view} endpoint failed: {e}")
            all_passed = False
    
    return all_passed

def main():
    """Run all tests."""
    print("Testing Prime Curve Flask Web Application...")
    print("=" * 50)
    
    tests = [
        test_main_page,
        test_plotly_js,
        test_plot_endpoints
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    if all(results):
        print("All tests passed! ✓")
        sys.exit(0)
    else:
        print("Some tests failed! ✗")
        sys.exit(1)

if __name__ == '__main__':
    main()